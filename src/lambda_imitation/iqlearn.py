"""SAC actor-critic with optional λ-discrepancy V-trace critics.

Implements a SAC-style actor-critic agent.  When ``approximate_lambda=True``,
two additional V-trace λ-return critics (one per λ value) are trained
alongside the standard SAC critic; their Huber discrepancy is added to the
joint loss as a regulariser (the "λ-discrepancy" line of work).

Networks share a single *recurrent feature extractor* (FE) — a projection
plus optional memory cell (RNN / GRU / LSTM / identity) — and a per-role
*head*.  The FE carry is threaded across env steps and reset on episode
boundaries; an R2D2-style burn-in warms the carry before BPTT begins.  Both
continuous (squashed Gaussian) and discrete (categorical, all-actions
critic) action spaces are supported via ``is_discrete``.

The twin-Q critic is implemented as two independent ``Head`` pairs grouped
in a :class:`TwinCriticState` NamedTuple, so a single optimizer operates on
both branches.

All state is immutable NamedTuples; the functional design
(``create_iqlearn`` factory + pure closures) keeps the loop compatible with
``jax.jit`` and ``jax.lax.scan``.  A single ``loss_combined`` forward pass
computes actor / critic / λ-critic / λ-discrepancy losses on one shared
unroll; ``jax.grad(..., argnums=[0,1,2,3,4])`` distributes gradients to FE
/ actor / critic / λ1 / λ2.

Typical usage::

    key = jax.random.key(42)
    key_fe, key_heads = jax.random.split(key)
    fe = RecurrentFeatureExtractor(obs_dim, projection_dim=128,
                                   memory_type="gru", memory_hidden_dim=128,
                                   rngs=nnx.Rngs(key_fe))
    state, fns = create_iqlearn(
        Hyperparameters(), buffer, action_dim, fe, key=key_heads,
        is_discrete=True, approximate_lambda=True,
    )
    state, env_state, metrics = fns.train(state, env, env_params, env_state, key)
    action, carry = fns.predict(state, obs, carry, deterministic=True)
"""

from functools import partial
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from .buffer import Buffer, create_buffer, create_sequence_sample

# Bounds for the squashed log-standard-deviation of the policy distribution.
# The raw output is tanh-squashed and then rescaled into this range to keep
# the distribution numerically stable while remaining expressive.
LOG_STD_MIN = -5
LOG_STD_MAX = 2


def cast_floating(tree, dtype: jnp.dtype):
    """Cast every floating-point leaf of a pytree to ``dtype``.

    Used to set the storage precision of a freshly-initialised parameter
    state.  Parameters are always *initialised* in float32 (some Flax cells
    use an orthogonal recurrent-kernel init whose QR step XLA does not support
    in bf16/fp16); this cast applies the requested storage dtype afterwards.
    Non-floating leaves (integer counters, booleans) are passed through
    unchanged, so it is safe to apply to whole graph states.

    Args:
        tree: Any JAX pytree (e.g. an ``nnx`` graph state).
        dtype: Target floating dtype.

    Returns:
        A pytree of the same structure with floating leaves cast to ``dtype``.
    """
    return jax.tree.map(
        lambda x: x.astype(dtype)
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
        else x,
        tree,
    )


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------


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
        dtype: Compute dtype for the linear layers (matmul / activation
            precision).  ``jnp.bfloat16`` / ``jnp.float16`` enable low-precision
            forward + backward passes.  Weights are always *initialised* in
            float32 (low-precision orthogonal/QR init is unsupported by XLA);
            storage precision is set afterwards by the caller via a param-state
            cast (see :func:`create_iqlearn`).
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        dims = [feature_dim] + list(hidden_dims) + [output_dim]
        self.layers = nnx.data(
            [
                nnx.Linear(dims[i], dims[i + 1], dtype=dtype, rngs=rngs)
                for i in range(len(dims) - 1)
            ]
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
    """Flax NNX graph states for a recurrent feature extractor.

    Both fields are ``nnx.GraphState`` objects produced by ``nnx.split``.
    Together they form a JAX pytree, so optimizer updates and EMA target
    updates work on them transparently via ``jax.tree.map``.

    Attributes:
        fe: Graph state of the projection / feature-extraction module.
        memory: Graph state of the recurrent memory cell (identity / RNN /
            GRU / LSTM).
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
    """Flax NNX graph definitions for a recurrent feature extractor.

    Static (non-parameter) graph descriptions produced by ``nnx.split`` and
    consumed by ``nnx.merge`` to reconstruct live modules during forward
    passes.

    Attributes:
        fe: Graph definition of the projection / feature-extraction module.
        memory: Graph definition of the recurrent memory cell.
    """

    fe: nnx.GraphDef
    memory: nnx.GraphDef


class TwinCriticGraph(NamedTuple):
    """Flax NNX graph definitions for both Q-branches of a twin critic.

    Attributes:
        q1: Graph definition of the first Q-branch head.
        q2: Graph definition of the second Q-branch head.
    """

    q1: nnx.GraphDef
    q2: nnx.GraphDef


class SACState(NamedTuple):
    """Complete, serialisable state of one SAC (+ optional λ-critic) agent.

    All fields are JAX pytrees, so the entire state can be checkpointed,
    passed through ``jax.jit``, or stacked for vectorised environments.
    The λ-critic fields (``lambda{1,2}_critic*``) are ``None`` when the
    agent was constructed with ``approximate_lambda=False``.

    Attributes:
        feature_extractor: Online recurrent FE state (projection + memory).
        actor: Online actor head state.
        critic: Online twin-critic state (two independent Q-branches).
        lambda1_critic: Online twin-critic trained with V-trace λ-return for
            ``params.lambda1``.  ``None`` when ``approximate_lambda=False``.
        lambda2_critic: Same, for ``params.lambda2``.
        feature_extractor_target: EMA copy of the FE.
        actor_target: EMA copy of the actor.
        critic_target: EMA copy of the critic.
        lambda1_critic_target: EMA copy of ``lambda1_critic``.
        lambda2_critic_target: EMA copy of ``lambda2_critic``.
        fe_optimizer_state: Optax state for the FE Adam optimiser.
        actor_optimizer_state: Optax state for the actor Adam optimiser.
        critic_optimizer_state: Optax state for the critic Adam optimiser;
            operates on the full :class:`TwinCriticState` pytree.
        lambda1_critic_optimizer_state: Optax state for the λ1-critic.
        lambda2_critic_optimizer_state: Optax state for the λ2-critic.
        alpha_optimizer_state: Optax state for the entropy temperature.
        alpha: Current entropy temperature (``exp(log_alpha)``).
        log_alpha: Log-space entropy temperature; directly optimised to avoid
            a positivity constraint.
        online_buffer: Circular replay buffer of online transitions.
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
    fe_optimizer_state: optax.OptState
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
        predict: ``(state, obs, carry, key, deterministic, ...) ->
            (action, new_carry[, extra])`` -- sample or compute a
            deterministic action for a single observation while threading
            the recurrent FE carry.  Discrete: returns a ``float32`` action
            index; ``return_prob=True`` additionally returns ``π(a|s)``.
            Continuous: returns a ``(action_dim,)`` action; ``return_unsquashed
            =True`` additionally returns the pre-tanh action.
        train: ``(state, env, env_params, env_state, key) ->
            (state, env_state, metrics)`` -- collect online transitions from
            a gymnax-compatible environment and run ``train_steps`` SAC
            gradient updates via ``jax.lax.scan``.  ``env`` is a static
            (non-traced) Python object; ``env_params`` / ``env_state`` are
            JAX pytrees.  If the online buffer is cold, :attr:`prefill_buffer`
            is invoked automatically before the JIT-compiled loop runs.
        get_importance_ratios: ``(actor, x, actions, behaviour_probs) ->
            ratios`` -- compute per-transition importance ratios
            ``π(a|s) / b(a|s)`` for a batch of ``(features, action)`` pairs
            collected under a behaviour policy ``b``.  ``x`` is **post-FE
            features**, not raw observations.  Discrete: ``actions`` are
            float32 indices.  Continuous: ``actions`` are the **unsquashed**
            (pre-tanh) values stored under ``unsquashed_action_key`` in the
            online buffer.  ``behaviour_probs`` are probabilities (discrete)
            or probability densities (continuous) under ``b``.
        prefill_buffer: ``(state, env, env_params, env_state, n_steps, key) ->
            (state, env_state)`` -- collect ``n_steps`` transitions using a
            uniform random policy and write them into the online buffer.
            Discrete: action ~ Uniform{0,…,action_dim-1}, ``behaviour_key =
            1/action_dim``.  Continuous: ``u ~ N(0, I)``, action squashed
            through tanh, ``behaviour_key = exp(log_prob)`` with the same
            change-of-variables as :attr:`get_importance_ratios`.  The last
            step is force-terminated so that all ``n_steps`` slots are
            immediately sampleable.  Called automatically by :attr:`train`
            when the online buffer has fewer than ``params.online_batch_size``
            sampleable transitions.
        train_unrolled: ``(state, env, env_params, env_state, env_carry, key)
            -> (state, env_state, env_carry, metrics)`` -- the pure, un-jitted
            ``train_steps``-long scan underlying :attr:`train`, without the
            host-side buffer warm-up check.  Safe to ``jax.vmap`` over a
            leading seed axis (buffer must be pre-filled first via
            :attr:`prefill_buffer`).
    """

    predict: Callable
    train: Callable
    get_importance_ratios: Callable
    prefill_buffer: Callable
    train_unrolled: Callable


class DebugFunctions(NamedTuple):
    """Optional debug helpers returned by :func:`create_iqlearn` when ``debug=True``.

    These functions expose internal computations that are useful for
    introspection and unit testing but are not required for normal training.
    All accept **post-FE features** (latents), not raw observations — callers
    must first run the FE to produce ``x``.

    Attributes:
        get_q: ``(critic, graph, x, actions) -> q_values`` -- evaluate the
            conservative (min-over-twin) Q-values for a batch of
            ``(features, action)`` pairs.  ``graph`` is the matching
            ``TwinCriticGraph``.  ``x`` has shape ``(batch, feature_dim)``;
            ``actions`` has shape ``(batch, action_dim)`` for continuous or
            ``(batch, 1)`` (float32 indices) for discrete.  Returns a float32
            array of shape ``(batch,)``.
        get_entropy: ``(actor, x, key) -> entropy`` -- per-state policy
            entropy under the current actor.  Returns a float32 array of
            shape ``(batch,)``.
    """

    get_q: Callable
    get_entropy: Callable


class Hyperparameters(NamedTuple):
    """Training hyperparameters.

    All fields have sensible defaults so callers only need to override what
    differs from the standard SAC setup.  λ-discrepancy fields
    (``lambda{1,2}``, ``lambda_critic_lr``, ``c_bar``, ``rho_bar``,
    ``lambda_truncation``, ``lambda_coef``, ``fake_onpolicy_loss``) are
    ignored unless ``approximate_lambda=True``.

    Attributes:
        fe_lr: Learning rate for the feature-extractor Adam optimiser.
        actor_lr: Learning rate for the actor Adam optimiser.
        critic_lr: Learning rate for the critic Adam optimiser.
        lambda_critic_lr: Learning rate for each λ-critic Adam optimiser.
        alpha_lr: Learning rate for the entropy temperature Adam optimiser.
        alpha: Initial entropy temperature.  When ``autotune_alpha=True``
            this is only the starting value.
        autotune_alpha: If True, alpha is continuously adjusted to match
            ``target_entropy``.  If False, alpha is held fixed.
        batch_size: Number of expert transitions sampled per gradient step
            (reserved for the IQ-Learn objective; currently unused by the
            SAC-only loss).
        gamma: Discount factor for future rewards.
        target_entropy: Desired policy entropy used by the alpha loss.  For
            continuous spaces a common heuristic is ``-action_dim``; for
            discrete spaces ``0.98 * log(num_actions)`` (Christodoulou 2019).
        online_buffer_size: Capacity of the circular online replay buffer.
        online_batch_size: Number of sequences sampled per gradient step.
            :func:`train` automatically pre-fills the buffer with at least
            this many random transitions on the first call.
        tau: Soft update coefficient for EMA target networks.
        lambda1: First V-trace λ value (typically near 0 — short horizon).
        lambda2: Second V-trace λ value (typically near 1 — long horizon).
        c_bar: Truncation cap for the V-trace ``c`` correction term.
        rho_bar: Truncation cap for the V-trace ``ρ`` TD-error weight.
        burn_in_length: Number of leading time-steps in each sampled sequence
            used to warm the recurrent FE carry; gradients are blocked at
            the burn-in / unroll boundary via ``stop_gradient``.
        sequence_length: Number of BPTT time-steps per sampled sequence
            (post burn-in, pre λ-truncation tail).
        lambda_truncation: Number of trailing time-steps dropped from the
            V-trace loss to avoid biasing the λ-return by missing bootstrap
            mass at the end of the unroll.  Each sampled sequence has total
            length ``burn_in_length + sequence_length + lambda_truncation``.
        lambda_coef: Multiplier on the Huber λ-discrepancy regulariser
            ``‖Q_λ1 − Q_λ2‖`` added to the joint loss.
        fake_onpolicy_loss: If True, set every V-trace importance ratio to
            1.0 (i.e. assume on-policy data).  Used as an ablation.
    """

    fe_lr: float = 1e-4
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    lambda_critic_lr: float = 1e-4
    alpha_lr: float = 1e-4
    alpha: float = 0.2
    autotune_alpha: bool = False
    batch_size: int = 256
    gamma: float = 0.99
    target_entropy: float = -1
    online_buffer_size: int = 10_000
    online_batch_size: int = 256
    tau: float = 0.005
    lambda1: float = 0.1
    lambda2: float = 0.9
    c_bar: float = 1.0
    rho_bar: float = 1.0
    burn_in_length: int = 20
    sequence_length: int = 80
    lambda_truncation: int = 30
    lambda_coef: float = 0.2
    fake_onpolicy_loss: bool = True


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
behaviour_key = "behaviour_weight"


def create_iqlearn(
    params: Hyperparameters,
    buffer: Buffer,
    action_dim: int,
    feature_extractor: nnx.Module,
    key: jax.Array,
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
    dtype: jnp.dtype = jnp.float32,
    param_dtype: jnp.dtype | None = None,
) -> (
    "Tuple[SACState, SACFunctions] | "
    "Tuple[SACState, SACFunctions, DebugFunctions]"
):
    """Construct a SAC (+ optional λ-discrepancy) agent.

    The recurrent feature extractor is taken as-is (already initialised by
    the caller), split into graph definition + parameter state via
    ``nnx.split``, and frozen inside the returned closures.  Actor / critic
    / (optional) λ-critic heads are created internally; their input
    dimension is inferred from a dummy forward pass through the FE.  The
    same FE is shared by all heads; each twin-Q critic still has two
    independent heads.

    The returned ``train`` function runs ``train_steps`` env-step + gradient
    iterations per call inside a single ``jax.lax.scan``, keeping the whole
    loop JIT-compiled after the first invocation.

    Args:
        params: Hyperparameters controlling learning rates, discount, alpha,
            burn-in length, λ values, etc.
        buffer: Reference replay buffer.  Used only to extract the obs /
            action shapes for the internal online buffer schema; its
            contents are not consumed.
        action_dim: Number of continuous action dimensions, or number of
            discrete actions when ``is_discrete=True``.
        feature_extractor: Initialised ``nnx.Module`` exposing the
            ``feature_extractor(carry, obs) -> (new_carry, y)`` calling
            convention and an ``initialize_carry(batch_size)`` helper (see
            ``utils.RecurrentFeatureExtractor``).  Ownership is transferred;
            do not use the module directly after this call.
        key: JAX PRNGKey used to initialise actor / critic / λ-critic
            head parameters.  Split internally so each head gets a
            unique sub-key.
        obs_key: Key in the online buffer that holds observations.
        action_key: Key in the online buffer that holds actions.
        action_scale: Per-dimension scale applied after the tanh squashing
            (continuous only).  Scalar or array of shape ``(action_dim,)``.
        action_bias: Per-dimension offset applied after the tanh squashing
            (continuous only).  Scalar or array of shape ``(action_dim,)``.
        train_steps: Number of (env-step + gradient) iterations per ``train``
            call.
        actor_dims: Hidden layer widths for the actor head.  Defaults to
            ``()`` (direct linear projection).
        critic_dims: Hidden layer widths for each SAC critic head.
        lambda1_critic_dims: Hidden layer widths for each λ1-critic head.
            Ignored when ``approximate_lambda=False``.
        lambda2_critic_dims: Same, for the λ2-critic head.
        is_discrete: If True, use a categorical actor and an all-actions
            critic; the soft value is computed exactly as
            ``Σ_a π(a|s)·(Q(s,a) − α·log π(a|s))``.  If False, use a
            squashed-Gaussian actor and a continuous critic that takes
            actions as additional input.
        approximate_lambda: If True, additionally train two V-trace
            λ-return critics (one per ``lambda{1,2}``) and add their Huber
            discrepancy to the joint loss as a regulariser.
        debug: If True, additionally return a :class:`DebugFunctions` named
            tuple exposing internal helpers (``get_q``, ``get_entropy``).
        dtype: Compute dtype for all actor / critic / λ-critic heads.  The
            feature extractor's dtype is fixed when it is constructed by the
            caller, so pass a matching ``dtype`` there too (see
            :func:`create_iqlearn_from_env`).  ``jnp.bfloat16`` is the safe
            low-precision choice; ``jnp.float16`` has a narrow range and may
            overflow without loss scaling.
        param_dtype: Storage dtype for ALL weights (FE + every head).  Applied
            as a post-init cast (see :func:`cast_floating`), so it also fixes
            the dtype of the Adam moments and the EMA target nets.  ``None``
            (default) ties it to ``dtype`` (full low-precision cast).  Setting
            ``param_dtype=jnp.float32`` while ``dtype`` is low precision gives
            mixed-precision (AMP-style) training: weights and optimizer moments
            stay fp32, only the matmuls run low-precision.

    Returns:
        When ``debug=False`` (default): a ``(SACState, SACFunctions)`` pair.
        When ``debug=True``: a ``(SACState, SACFunctions, DebugFunctions)``
        triple.  ``SACState`` is the initial agent state with online and
        target networks set to the same weights; ``SACFunctions`` bundles
        ``predict`` / ``train`` / ``get_importance_ratios`` / ``prefill_buffer``.
    """
    # Online buffer: same obs/action shapes as the reference buffer, plus
    # scalar reward / terminated / (optionally) behaviour-prob fields written
    # by run_env_step.
    online_shapes = {
        **extract_buffer_shapes(buffer),
        reward_key: (),
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
    online_mc_this_keys = [
        obs_key,
        action_key,
        reward_key,
        terminated_key,
        behaviour_key,
    ]
    online_buffer_lambda_sample = create_sequence_sample(
        online_buffer.size,
        params.online_batch_size,
        params.burn_in_length + params.sequence_length + params.lambda_truncation,
        online_mc_this_keys,
    )
    # Pre-initialise every behaviour_key slot to 1.0 so that slots which
    # have never been written by run_env_step (pre-fill and as-yet-unwritten
    # slots) contribute an IS denominator of 1.0 rather than 0.0.
    # Division by zero would otherwise produce NaN in the V-trace loss.
    # Real transitions overwrite their slot with the true policy probability.
    online_buffer = online_buffer._replace(
        info={
            **online_buffer.info,
            behaviour_key: jnp.ones_like(online_buffer.info[behaviour_key]),
        }
    )

    # Infer feature dims via dummy forward pass (before split).
    # FE call returns (new_carry, y); we only need y's last dim.
    dummy_obs = jnp.zeros((1,) + buffer.info[obs_key].shape[1:])
    _, dummy_y = feature_extractor(
        feature_extractor.initialize_carry(1), dummy_obs
    )
    feature_dim = dummy_y.shape[-1]

    # Create heads — discrete and continuous differ only in output_dim and
    # whether actions are concatenated to features before the head.
    # Split key so each head gets unique initialisation.
    k_actor, k_cq1, k_cq2, k_l1q1, k_l1q2, k_l2q1, k_l2q2 = \
        jax.random.split(key, 7)
    if is_discrete:
        actor_model = Head(
            feature_dim,
            actor_dims,
            action_dim,
            dtype=dtype,
            rngs=nnx.Rngs(k_actor),
        )
        critic_q1_model = Head(
            feature_dim,
            critic_dims,
            action_dim,
            dtype=dtype,
            rngs=nnx.Rngs(k_cq1),
        )
        critic_q2_model = Head(
            feature_dim,
            critic_dims,
            action_dim,
            dtype=dtype,
            rngs=nnx.Rngs(k_cq2),
        )
        if approximate_lambda:
            lambda1_q1_critic_model = Head(
                feature_dim,
                lambda1_critic_dims,
                action_dim,
                dtype=dtype,
                rngs=nnx.Rngs(k_l1q1),
            )
            lambda1_q2_critic_model = Head(
                feature_dim,
                lambda1_critic_dims,
                action_dim,
                dtype=dtype,
                rngs=nnx.Rngs(k_l1q2),
            )
            lambda2_q1_critic_model = Head(
                feature_dim,
                lambda2_critic_dims,
                action_dim,
                dtype=dtype,
                rngs=nnx.Rngs(k_l2q1),
            )
            lambda2_q2_critic_model = Head(
                feature_dim,
                lambda2_critic_dims,
                action_dim,
                dtype=dtype,
                rngs=nnx.Rngs(k_l2q2),
            )
    else:
        actor_model = Head(
            feature_dim,
            actor_dims,
            2 * action_dim,
            dtype=dtype,
            rngs=nnx.Rngs(k_actor),
        )
        # For continuous critics, features and actions are concatenated before
        # the head, so input_dim = feature_dim + action_dim, output_dim = 1.
        critic_q1_model = Head(
            feature_dim + action_dim,
            critic_dims,
            1,
            dtype=dtype,
            rngs=nnx.Rngs(k_cq1),
        )
        critic_q2_model = Head(
            feature_dim + action_dim,
            critic_dims,
            1,
            dtype=dtype,
            rngs=nnx.Rngs(k_cq2),
        )
        if approximate_lambda:
            lambda1_q1_critic_model = Head(
                feature_dim + action_dim,
                lambda1_critic_dims,
                1,
                dtype=dtype,
                rngs=nnx.Rngs(k_l1q1),
            )
            lambda1_q2_critic_model = Head(
                feature_dim + action_dim,
                lambda1_critic_dims,
                1,
                dtype=dtype,
                rngs=nnx.Rngs(k_l1q2),
            )
            lambda2_q1_critic_model = Head(
                feature_dim + action_dim,
                lambda2_critic_dims,
                1,
                dtype=dtype,
                rngs=nnx.Rngs(k_l2q1),
            )
            lambda2_q2_critic_model = Head(
                feature_dim + action_dim,
                lambda2_critic_dims,
                1,
                dtype=dtype,
                rngs=nnx.Rngs(k_l2q2),
            )

    # Split all six modules into (graph_def, state)
    feature_extractor_graph, feature_extractor_state = nnx.split(feature_extractor)
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

    # Apply the global storage precision.  ``param_dtype=None`` ties storage to
    # the compute dtype (full low-precision cast); passing ``jnp.float32`` while
    # ``dtype`` is bf16/fp16 yields mixed-precision (AMP) training where weights
    # and Adam moments stay fp32 but matmuls run low-precision.  Casting here —
    # before optimizer init and before the EMA targets are copied below — keeps
    # the optimizer moments and target nets in the same storage dtype.
    storage_dtype = dtype if param_dtype is None else param_dtype
    feature_extractor_state = cast_floating(feature_extractor_state, storage_dtype)
    actor_state = cast_floating(actor_state, storage_dtype)
    critic_state = cast_floating(critic_state, storage_dtype)
    if approximate_lambda:
        lambda1_critic_state = cast_floating(lambda1_critic_state, storage_dtype)
        lambda2_critic_state = cast_floating(lambda2_critic_state, storage_dtype)

    # Optimizers: actor operates on its nnx.GraphState; critic on TwinCriticState.
    fe_optimizer = optax.adam(params.fe_lr)
    actor_optimizer = optax.adam(params.actor_lr)
    critic_optimizer = optax.adam(params.critic_lr)
    if approximate_lambda:
        lambda1_critic_optimizer = optax.adam(params.lambda_critic_lr)
        lambda2_critic_optimizer = optax.adam(params.lambda_critic_lr)
    alpha_optimizer = optax.adam(params.alpha_lr)

    log_alpha = jnp.array(jnp.log(params.alpha))
    fe_optimizer_state = fe_optimizer.init(feature_extractor_state)
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
        remove_weak_types(fe_optimizer_state),
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
        """Reconstruct and run the actor head on pre-extracted features x."""
        head = nnx.merge(actor_graph, actor)
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
            """Per-branch Q-values for the taken action in each transition.

            Args:
                critic: Twin-critic network state.
                graph: Matching twin-critic graph definition.
                x: Feature batch of shape ``(batch, feature_dim)``.
                actions: Float32 array of action indices, shape
                    ``(batch,)`` or ``(batch, 1)`` (e.g. 0.0, 1.0, 2.0).

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
            """Conservative (min over twin) Q-value for each transition.

            Args:
                critic: Twin-critic network state.
                graph: Matching twin-critic graph definition.
                x: Feature batch of shape ``(batch, feature_dim)``.
                actions: Float32 array of action indices, shape
                    ``(batch,)`` or ``(batch, 1)``.

            Returns:
                Per-transition Q-value of shape ``(batch,)``.
            """
            q1, q2 = get_q_both(critic, graph, x, actions)
            return jnp.minimum(q1, q2)

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
                obs: Single observation of shape ``(*obs_shape,)`` (no batch
                    dim).
                carry: Recurrent FE carry of shape ``(carry_dim,)`` from the
                    previous step (zero at episode start).
                key: JAX PRNG key, used only when ``deterministic=False``.
                deterministic: If True, return the greedy (argmax) action.
                    If False, sample from the categorical policy.
                return_prob: If True, also return ``π(a|s)`` for the selected
                    action.

            Returns:
                ``(action, new_carry)``, or ``(action, new_carry, prob)`` when
                ``return_prob=True``.  ``action`` is a ``float32`` index.
            """
            obs_batch = jnp.expand_dims(obs, 0)
            carry_batch = jnp.expand_dims(carry, 0)
            new_carry, x = nnx.merge(
                feature_extractor_graph, iqlearn.feature_extractor
            )(carry_batch, obs_batch)
            logits = run_actor(iqlearn.actor, x)[0]  # (num_actions,)
            if deterministic:
                action = jnp.argmax(logits)
            else:
                action = jax.random.categorical(key, logits)
            if return_prob:
                return (
                    action.astype(jnp.float32),
                    new_carry[0],
                    jax.nn.softmax(logits)[action],
                )
            return (
                action.astype(jnp.float32),
                new_carry[0],
            )

        @jax.jit
        def get_importance_ratios(
            actor: nnx.GraphState,
            x: jax.Array,
            actions: jax.Array,
            behaviour_probs: jax.Array,
        ) -> jax.Array:
            """Compute importance ratios ``π(a|s) / b(a|s)`` for a batch.

            Supports any leading batch dims (e.g. ``(B,)`` or ``(B, T)``);
            shapes are inferred from the actor output.

            Args:
                actor: Current actor network state.
                x: Feature batch of shape ``(*batch_dims, feature_dim)``.
                actions: Integer action indices stored as float32, shape
                    ``(*batch_dims,)`` or ``(*batch_dims, 1)``.
                behaviour_probs: Behaviour-policy probabilities, shape
                    broadcastable to ``(*batch_dims,)``.

            Returns:
                Importance ratios of shape ``(*batch_dims,)``.
            """
            logits = run_actor(actor, x)  # (*batch_dims, num_actions)
            probs = jax.nn.softmax(logits)
            idx = jnp.round(actions).astype(jnp.int32).reshape(probs.shape[:-1] + (1,))
            pi_a = jnp.take_along_axis(probs, idx, axis=-1).squeeze(-1)
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

            Supports any leading batch dims (e.g. ``(B,)`` or ``(B, T)``);
            log-prob sums over the trailing ``action_dim`` axis only.

            Args:
                actor: Current actor network state.
                x: Feature batch of shape ``(*batch_dims, feature_dim)``.
                actions: **Unsquashed** (pre-tanh) action values, shape
                    ``(*batch_dims, action_dim)``.
                behaviour_probs: Behaviour-policy densities, shape
                    broadcastable to ``(*batch_dims,)``.

            Returns:
                Importance ratios of shape ``(*batch_dims,)``.
            """
            mean, std = get_dist_params(actor, x)  # (*batch_dims, action_dim)
            u = actions  # pre-tanh, no inversion needed
            y_t = jnp.tanh(u)
            log_prob = (
                -((u - mean) ** 2) / (2 * std**2)
                - 0.5 * jnp.log(2 * jnp.pi)
                - jnp.log(std)
                - jnp.log(action_scale * (1 - y_t**2) + 1e-6)
            ).sum(
                axis=-1
            )  # (*batch_dims,)
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
            """Compute a continuous action for a single observation.

            Args:
                iqlearn: Current agent state.
                obs: Single observation of shape ``(*obs_shape,)`` (no batch
                    dim).
                carry: Recurrent FE carry of shape ``(carry_dim,)`` from the
                    previous step (zero at episode start).
                key: JAX PRNG key, used only when ``deterministic=False``.
                deterministic: If True, return the tanh-squashed policy mean
                    (no sampling noise).  If False, sample from the full
                    Gaussian policy.
                return_unsquashed: If True, also return the pre-tanh action
                    (needed for stored behaviour probabilities).

            Returns:
                ``(action, new_carry)``, or
                ``(action, new_carry, unsquashed_action)`` when
                ``return_unsquashed=True``.  ``action`` has shape
                ``(action_dim,)`` and is scaled / shifted by ``action_scale``
                and ``action_bias``.
            """
            obs = jnp.expand_dims(obs, 0)
            carry = jnp.expand_dims(carry, 0)
            new_carry, x = nnx.merge(
                feature_extractor_graph, iqlearn.feature_extractor
            )(carry, obs)
            mean, std = get_dist_params(iqlearn.actor, x)
            if deterministic:
                unsquashed_action = mean
            else:
                unsquashed_action = jax.random.normal(key, mean.shape) * std + mean
            y_t = jnp.tanh(unsquashed_action)
            action = y_t * action_scale + action_bias
            if return_unsquashed:
                return action[0], new_carry[0], unsquashed_action[0]
            return action[0], new_carry[0]

    # ------------------------------------------------------------------
    # Loss functions and other helpers which are structurally identical
    # for both action space types; differences are absorbed by the
    # action-space-specific helpers above.
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
        actor: nnx.GraphState,
        critic: TwinCriticState,
        latents: jax.Array,
        alpha: jax.Array,
        key: jax.Array,
    ) -> Tuple[jax.Array, dict]:
        """Actor loss: maximise the soft state value V(s) = Q(s,a) - α log π(a|s).

        Receives pre-computed latents (post-FE features); the caller is
        responsible for wrapping the online critic in
        ``jax.lax.stop_gradient`` so the actor gradient does not train the
        critic head (gradient still flows into the shared FE via ``latents``).

        Args:
            actor: Actor head state.
            critic: (Stop-gradient'd) online twin-critic state.
            latents: Features of shape ``(batch, feature_dim)``.
            alpha: Current entropy temperature.
            key: JAX PRNG key (used only on the continuous path).

        Returns:
            ``(scalar_loss, metrics)`` where metrics contains ``"q"``,
            ``"entropy"`` and ``"v"``.
        """
        key_v = key
        v, metrics = get_v(
            actor,
            critic,
            critic_graph,
            alpha,
            latents,
            key_v,
            include_entropy=True,
            include_log=True,
        )
        metrics.update({"v": v.mean()})
        return -v.mean(), metrics

    def loss_critic(
        actor_target: nnx.GraphState,
        critic: TwinCriticState,
        critic_target: TwinCriticState,
        latents: jax.Array,
        target_latents: jax.Array,
        actions: jax.Array,
        rewards: jax.Array,
        terminated: jax.Array,
        alpha: jax.Array,
        key: jax.Array,
    ) -> Tuple[jax.Array, dict]:
        """SAC Bellman MSE loss for the twin-critic (continuous and discrete).

        Computes independent TD errors for both Q-branches against the shared
        target ``r + γ(1−done)·V(s')``.  ``V(s')`` is computed under the
        target actor and target critic; for discrete spaces this is the exact
        closed-form inner product, for continuous spaces it uses a sampled
        action.

        Args:
            actor_target: EMA-smoothed actor used to compute ``V(s')``.
            critic: Online twin-critic being optimised.
            critic_target: EMA-smoothed critic used inside ``V(s')``.
            latents: Features of state ``s``, shape ``(batch, feature_dim)``.
            target_latents: Target-FE features of state ``s'``, shape
                ``(batch, feature_dim)``.
            actions: Action taken at ``s`` — float32 indices ``(batch, 1)``
                for discrete or ``(batch, action_dim)`` for continuous.
            rewards: Per-step rewards, shape ``(batch,)``.
            terminated: Per-step done flags (float32 0/1), shape ``(batch,)``.
            alpha: Current entropy temperature.
            key: JAX PRNG key (used only on the continuous path).

        Returns:
            ``(scalar_loss, metrics)`` where metrics contains
            ``"critic_loss"`` and ``"target_q"``.
        """
        key_v = key

        next_v = get_v(
            actor_target,
            critic_target,
            critic_graph,
            alpha,
            target_latents,
            key_v,
            include_entropy=True,
        )
        target_q = jax.lax.stop_gradient(
            rewards + params.gamma * (1.0 - terminated) * next_v
        )

        q1, q2 = get_q_both(critic, critic_graph, latents, actions)
        loss = 0.5 * (jnp.mean((q1 - target_q) ** 2) + jnp.mean((q2 - target_q) ** 2))
        return loss, {"critic_loss": loss, "target_q": target_q.mean()}

    def loss_ld(
        lambda1_critic: TwinCriticState,
        lambda2_critic: TwinCriticState,
        lambda1_graph: TwinCriticGraph,
        lambda2_graph: TwinCriticGraph,
        latents: jax.Array,
        actions: jax.Array,
    ) -> Tuple[jax.Array, dict]:
        """λ-discrepancy regulariser pulling the two λ-critics together.

        Computes the Huber loss between the conservative (min-over-twin)
        Q-values of the short-horizon (``params.lambda1``) and long-horizon
        (``params.lambda2``) V-trace critics evaluated at the same
        ``(s, a)``.  Added to the joint loss with weight
        ``params.lambda_coef``.

        Args:
            lambda1_critic: Online λ1 twin-critic state.
            lambda2_critic: Online λ2 twin-critic state.
            lambda1_graph: Graph for ``lambda1_critic``.
            lambda2_graph: Graph for ``lambda2_critic``.
            latents: Features of shape ``(batch, feature_dim)``.
            actions: Action taken at each state.

        Returns:
            ``(scalar_loss, {"ld_loss": loss})``.
        """

        q1 = get_q(lambda1_critic, lambda1_graph, latents, actions)
        q2 = get_q(lambda2_critic, lambda2_graph, latents, actions)
        loss = optax.losses.huber_loss(q1, q2).mean()
        return loss, {"ld_loss": loss}

    # ------------------------------------------------------------------
    # Online helpers: environment interaction and SAC update
    # ------------------------------------------------------------------

    def run_env_step(
        sac: SACState, env, env_params, env_state, env_carry, key: jax.Array
    ):
        """Collect one transition from a gymnax environment into the online buffer.

        Threads the recurrent FE carry across env steps: the carry from the
        previous call is fed back into ``predict`` so the policy sees the full
        rollout history.  On episode termination the carry is reset to zero
        (matching the per-episode reset performed during sequence training).

        Args:
            sac: Current agent state.  Only ``sac.online_buffer`` is mutated.
            env: Gymnax environment object (static — not traced by JAX).
            env_params: Gymnax environment parameters pytree.
            env_state: Current gymnax environment state pytree.
            env_carry: FE carry of shape ``(carry_dim,)`` from the previous
                step (zero on the first step of an episode).
            key: JAX PRNG key; split internally for action sampling and env step.

        Returns:
            ``(new_sac, new_env_state, new_carry)`` where ``new_carry`` is the
            post-step FE carry (reset to zero on episode termination).
        """
        key_act, key_step = jax.random.split(key, 2)
        obs = env.get_obs(env_state, env_params)
        if is_discrete:
            action, new_carry, prob = predict(
                sac, obs, env_carry, key_act, return_prob=True
            )
            env_action = jnp.round(action).astype(jnp.int32)
        else:
            if approximate_lambda:
                action, new_carry, unsquashed_action = predict(
                    sac, obs, env_carry, key_act, return_unsquashed=True
                )
                # behaviour probability density under the squashed Gaussian
                _, logprob = sample_action_logprob(sac.actor, obs, key_act)[1:3]
                prob = jnp.exp(logprob)
            else:
                action, new_carry = predict(sac, obs, env_carry, key_act)
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
        new_carry = jax.lax.select(done, jnp.zeros_like(new_carry), new_carry)
        return (
            sac._replace(online_buffer=new_online_buffer),
            new_env_state,
            new_carry,
        )

    def calculate_latent(
        feature_extractor_state: nnx.GraphState,
        target_feature_extractor_state: nnx.GraphState,
        observations: jax.Array,
        dones: jax.Array,
        init_carries: jax.Array,
    ):
        """Roll the online and target FEs over a sampled sequence.

        Performs an R2D2-style burn-in over the first ``params.burn_in_length``
        steps to warm the recurrent carry (gradients stopped at the burn-in
        boundary) and then unrolls the remainder to produce per-step latents.
        The carry is reset to zeros at any step where ``dones=True``.

        Online and target FEs share the same scan so they see the same carry
        reset pattern; both carries are initialised from ``init_carries``.

        Args:
            feature_extractor_state: Online FE state (FE + memory).
            target_feature_extractor_state: Target FE state.
            observations: Batch-major observations of shape ``(B, T, *obs)``.
            dones: Per-step termination flags, shape ``(B, T)``.
            init_carries: Initial carries of shape ``(B, carry_dim)`` (zero
                for the start of each train() call).

        Returns:
            ``(latent, target_latent)``, both time-major with shape
            ``(T - burn_in_length, B, feature_dim)``.
        """
        feature_extractor = nnx.merge(feature_extractor_graph, feature_extractor_state)
        target_feature_extractor = nnx.merge(
            feature_extractor_graph, target_feature_extractor_state
        )
        _BL = params.burn_in_length

        # Sample layout is (B, T, ...); lax.scan iterates leading axis, so move
        # the time axis to the front for the per-step inputs.
        observations = jnp.swapaxes(observations, 0, 1)
        dones = jnp.swapaxes(dones, 0, 1)

        carry_reset = jax.vmap(
            lambda done, carry: jax.lax.select(done, jnp.zeros_like(carry), carry)
        )

        def scan_carries(scan_carry, x):
            carry, target_carry = scan_carry
            obs, done_step = x
            done_step = done_step.astype(jnp.bool_)
            new_carry, y = feature_extractor(carry, obs)
            new_target_carry, y_target = target_feature_extractor(target_carry, obs)
            new_carry = carry_reset(done_step, new_carry)
            new_target_carry = carry_reset(done_step, new_target_carry)

            return (new_carry, new_target_carry), (y, y_target)

        # Pair online + target carries; both start from the same zero tensor.
        burnt_in_carries, _ = jax.lax.scan(
            scan_carries,
            (init_carries, init_carries),
            (observations[:_BL], dones[:_BL]),
        )
        burnt_in_carries = jax.lax.stop_gradient(burnt_in_carries)

        _, latent = jax.lax.scan(
            scan_carries,
            burnt_in_carries,
            (observations[_BL:], dones[_BL:]),
        )

        return latent

    def loss_vtrace_lambda_sequence(
        target_actor_state: nnx.GraphState,
        q_state: TwinCriticState,
        q_target_state: TwinCriticState,
        q_graph: TwinCriticGraph,
        lam: float,
        actions: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        behaviour_probs: jax.Array,
        latents: jax.Array,
        target_latents: jax.Array,
        key: jax.Array,
    ):
        """V-trace λ-return Huber loss for one λ-critic over a time-major sequence.

        Inputs are time-major (``T - burn_in_length`` along axis 0).  Per-step
        importance ratios ``π_target(a|s) / b(a|s)`` are recomputed inline
        from the rolled features (no separate raw-obs IS pass).  When
        ``params.fake_onpolicy_loss`` is set the ratios are clamped to 1.

        The V-trace recursion uses ``(1 - done)`` to mask both the
        ``δ_V = r + γ(1-done) V_{s+1} - V_s`` term and the correction
        ``(1-done) γ c (v_{s+1} - V_{s+1})``.  The last
        ``params.lambda_truncation`` time-steps are dropped from the loss to
        avoid biasing the λ-return by missing bootstrap mass at the unroll
        tail.

        Currently asserts ``is_discrete=True``; continuous V-trace would
        require unsquashed (pre-tanh) actions plumbed through, which is not
        done yet.

        Args:
            target_actor_state: EMA actor used to compute ``V(s)`` and IS
                numerators.
            q_state: Online λ-critic being optimised.
            q_target_state: EMA λ-critic used inside ``V(s)``.
            q_graph: Graph for the λ-critic.
            lam: λ value (``params.lambda1`` or ``params.lambda2``); used as
                the ``c`` truncation factor and as a metric tag.
            actions: Time-major actions, shape ``(T', B, ...)``.
            rewards: Time-major rewards, shape ``(T', B)``.
            dones: Time-major termination flags, shape ``(T', B)``.
            behaviour_probs: Time-major behaviour-policy probabilities
                (or densities), shape ``(T', B)``.
            latents: Online-FE features, shape ``(T', B, feature_dim)``.
            target_latents: Target-FE features, shape ``(T', B, feature_dim)``.
            key: JAX PRNG key, threaded through the inner scan for sampled
                ``V(s)`` (unused on the discrete path).

        Returns:
            ``(scalar_loss, metrics)``.
        """
        assert is_discrete, (
            "loss_vtrace_lambda_sequence does not yet support continuous action "
            "spaces: importance ratios would require unsquashed (pre-tanh) actions, "
            "which are not currently wired through."
        )

        def scan_v_q(scan_carry, x):
            key = scan_carry
            key, key_next = jax.random.split(key)
            x, target_x, action, beh_prob = x
            v = get_v(
                target_actor_state,
                q_target_state,
                q_graph,
                jnp.array(0.0),
                target_x,
                key,
                False,
                False,
            )
            q = get_q(q_state, q_graph, x, action)
            if params.fake_onpolicy_loss:
                ratio = jnp.ones_like(beh_prob)
            else:
                ratio = get_importance_ratios(
                    target_actor_state, target_x, action, beh_prob
                )
            new_scan_carry = key_next

            return new_scan_carry, (v, q, ratio)

        # other way round for recursive definition of V-trace targets
        def scan_target(scan_carry, x):
            v_sp1, V_sp1 = scan_carry
            V_s, done, reward, ratio = x
            rho = jnp.minimum(params.rho_bar, ratio)
            c = lam * jnp.minimum(params.c_bar, ratio)
            delta_V = reward + (1 - done) * params.gamma * V_sp1 - V_s
            v_s = V_s + rho * delta_V + (1 - done) * params.gamma * c * (v_sp1 - V_sp1)
            return (v_s, V_s), v_s

        _carry, (v, q, ratios) = jax.lax.scan(
            scan_v_q,
            key,
            (
                latents,
                target_latents,
                actions,
                behaviour_probs,
            ),
        )

        # Run the V-trace recursion in float32.  ``v`` may be low precision
        # (it comes from a bf16/fp16 critic) but the recursion mixes in float32
        # rewards / dones, which would promote the scan carry mid-loop and break
        # lax.scan's in==out dtype requirement.  Value bootstrapping is also the
        # most precision-sensitive part of the loss, so fp32 here is the safe
        # choice; ``q`` stays in compute dtype and the Huber loss promotes.
        v = v.astype(jnp.float32)

        _carry, targets = jax.lax.scan(
            scan_target,
            (jnp.zeros_like(v[0]), jnp.zeros_like(v[0])),
            (v, dones.astype(jnp.float32), rewards.astype(jnp.float32), ratios),
            reverse=True,
        )

        loss = optax.losses.huber_loss(
            q[: -params.lambda_truncation],
            jax.lax.stop_gradient(targets)[: -params.lambda_truncation],
        ).mean()
        metrics = {
            "loss": loss,
            f"lambda{lam}_critic:": q[: -params.lambda_truncation].mean(),
            f"lambda{lam}_target:": targets[: -params.lambda_truncation].mean(),
        }

        return loss, metrics

    def loss_combined(
        feature_extractor_state,
        actor_state,
        critic_state,
        lambda1_critic_state,
        lambda2_critic_state,
        target_feature_extractor_state,
        actor_target_state,
        critic_target_state,
        lambda1_critic_target_state,
        lambda2_critic_target_state,
        alpha,
        buffer: Buffer,
        key: jax.Array,
    ):
        """Joint R2D2-style sequence loss: actor + critic + (optionally) λ-critics.

        Samples one batch of contiguous sequences from the online buffer,
        runs the recurrent FE (with burn-in) over each sequence under both
        the online and target weights, and computes:

        - **Actor loss** on flattened ``(t, b)`` features, with the online
          critic wrapped in ``stop_gradient`` so the actor gradient does not
          train the critic head (gradient still flows into the shared FE).
        - **Critic loss** with the off-by-one bootstrap pair
          ``latent[t] → target_latent[t+1]``, action/reward/terminated at
          index ``t``.
        - **λ-critic V-trace losses** (when ``approximate_lambda=True``), one
          per λ value, plus a Huber λ-discrepancy term ``loss_ld`` scaled by
          ``params.lambda_coef``.

        ``jax.grad(loss_combined, argnums=[0,1,2,3,4])`` distributes
        gradients to the FE / actor / critic / λ1-critic / λ2-critic
        respectively; target nets are passed positionally outside ``argnums``
        so JAX treats them as constants (no leakage).

        Returns:
            ``(scalar_loss, metrics_dict)``.
        """
        _BL = params.burn_in_length

        key_sample, key_actor, key_critic, key_lambda_critic1, key_lambda_critic2 = (
            jax.random.split(key, 5)
        )
        sample, indices = online_buffer_lambda_sample(buffer, key_sample)
        init_carries = feature_extractor.initialize_carry(params.online_batch_size)

        observations = sample.this_info[obs_key]
        actions = sample.this_info[action_key]
        rewards = sample.this_info[reward_key]
        terminated = sample.this_info[terminated_key]
        behaviour = sample.this_info[behaviour_key]

        latent, target_latent = calculate_latent(
            feature_extractor_state,
            target_feature_extractor_state,
            observations,
            terminated,
            init_carries,
        )
        # latent / target_latent shapes: (T - _BL, B, feat) -- time-major.

        # Sample fields are (B, T, ...); convert to time-major and slice down to
        # the BPTT (post-burn-in) window so they align with the latents.
        actions_tm = jnp.swapaxes(actions, 0, 1)[_BL:]
        rewards_tm = jnp.swapaxes(rewards, 0, 1)[_BL:]
        terminated_tm = jnp.swapaxes(terminated, 0, 1)[_BL:]
        behaviour_tm = jnp.swapaxes(behaviour, 0, 1)[_BL:]

        def _flat(x):
            # (T', B, ...) -> (T' * B, ...)
            return x.reshape((-1, *x.shape[2:]))

        # Actor: per-state objective, treat (t, b) as an independent batch.
        # Use the *online* critic but freeze its params via stop_gradient so
        # the actor loss does not train the critic. Gradient still flows into
        # the FE via the latent input, which is intended for the shared encoder.
        l_actor, metrics = loss_actor(
            actor_state,
            jax.lax.stop_gradient(critic_state),
            _flat(latent),
            alpha,
            key_actor,
        )

        # Critic: pair (latent[t], target_latent[t+1]) so V(s') is computed at
        # the next state. Action / reward / terminated at index t.
        l_critic, metrics_critic = loss_critic(
            actor_target_state,
            critic_state,
            critic_target_state,
            _flat(latent[:-1]),
            _flat(target_latent[1:]),
            _flat(actions_tm[:-1]),
            rewards_tm[:-1].reshape(-1),
            terminated_tm[:-1].reshape(-1),
            alpha,
            key_critic,
        )
        metrics.update(metrics_critic)

        loss = l_actor + l_critic
        if approximate_lambda:
            l_lambda1, metrics_lambda1_critic = loss_vtrace_lambda_sequence(
                actor_target_state,
                lambda1_critic_state,
                lambda1_critic_target_state,
                lambda1_critic_graph,
                params.lambda1,
                actions_tm,
                rewards_tm,
                terminated_tm,
                behaviour_tm,
                latent,
                target_latent,
                key_lambda_critic1,
            )
            metrics.update(metrics_lambda1_critic)
            l_lambda2, metrics_lambda2_critic = loss_vtrace_lambda_sequence(
                actor_target_state,
                lambda2_critic_state,
                lambda2_critic_target_state,
                lambda2_critic_graph,
                params.lambda2,
                actions_tm,
                rewards_tm,
                terminated_tm,
                behaviour_tm,
                latent,
                target_latent,
                key_lambda_critic2,
            )
            metrics.update(metrics_lambda2_critic)

            l_ld, metrics_ld = loss_ld(
                lambda1_critic_state,
                lambda2_critic_state,
                lambda1_critic_graph,
                lambda2_critic_graph,
                _flat(latent[:-1]),
                _flat(actions_tm[:-1]),
            )
            metrics.update(metrics_ld)
            loss += l_lambda1 + l_lambda2 + params.lambda_coef* l_ld

        return loss, metrics

    def update_step(sac: SACState, key: jax.Array) -> Tuple[SACState, dict]:
        """Execute one joint gradient step against :func:`loss_combined`.

        Computes a single ``jax.grad(loss_combined, argnums=[0,1,2,3,4])``
        and distributes the resulting gradients to the FE / actor / critic /
        λ1-critic / λ2-critic optimizers.  Also runs the optional alpha
        update (when ``params.autotune_alpha``) and the EMA target updates
        for all networks.  The online buffer must already hold at least
        ``params.online_batch_size`` sampleable transitions (guaranteed by
        :func:`prefill_buffer`, which :func:`train` calls automatically when
        the buffer is cold).

        Args:
            sac: Current agent state.
            key: JAX PRNG key; split internally inside ``loss_combined`` for
                sequence sampling and the actor / critic / λ-critic losses.

        Returns:
            ``(new_state, metrics)`` where metrics includes ``"q"``,
            ``"entropy"``, ``"v"``, ``"critic_loss"``, ``"target_q"``,
            (when ``approximate_lambda``) ``"ld_loss"`` and the V-trace
            per-λ tags, and ``"alpha"`` (when ``params.autotune_alpha``).
        """

        (
            grads_fe,
            grads_actor,
            grads_critic,
            grads_lambda1_critic,
            grads_lambda2_critic,
        ), metrics = jax.grad(loss_combined, argnums=[0, 1, 2, 3, 4], has_aux=True)(
            sac.feature_extractor,
            sac.actor,
            sac.critic,
            sac.lambda1_critic,
            sac.lambda2_critic,
            sac.feature_extractor_target,
            sac.actor_target,
            sac.critic_target,
            sac.lambda1_critic_target,
            sac.lambda2_critic_target,
            sac.alpha,
            sac.online_buffer,
            key,
        )

        # Optax's Adam bias-correction divides the (low-precision) moments by
        # float32 scalars, so both the updates and the resulting parameters are
        # promoted to float32.  Under jax.lax.scan the carry dtype must be
        # identical in and out, so when params are stored in a low-precision
        # ``storage_dtype`` we cast the new params *and* optimizer moments back
        # to it after every step.  For float32 storage these casts are no-ops.
        def _apply(opt_update_fn, params, grads, opt_state):
            updates, new_opt = opt_update_fn(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return (
                cast_floating(new_params, storage_dtype),
                cast_floating(new_opt, storage_dtype),
            )

        new_fe, new_fe_opt = _apply(
            fe_optimizer.update, sac.feature_extractor, grads_fe,
            sac.fe_optimizer_state,
        )
        new_actor, new_actor_opt = _apply(
            actor_optimizer.update, sac.actor, grads_actor,
            sac.actor_optimizer_state,
        )
        new_critic, new_critic_opt = _apply(
            critic_optimizer.update, sac.critic, grads_critic,
            sac.critic_optimizer_state,
        )

        if approximate_lambda:
            new_lambda1_critic, new_lambda1_critic_opt = _apply(
                lambda1_critic_optimizer.update, sac.lambda1_critic,
                grads_lambda1_critic, sac.lambda1_critic_optimizer_state,
            )
            new_lambda2_critic, new_lambda2_critic_opt = _apply(
                lambda2_critic_optimizer.update, sac.lambda2_critic,
                grads_lambda2_critic, sac.lambda2_critic_optimizer_state,
            )

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

        new_fe_target = jax.tree.map(
            lambda x, y: (1 - params.tau) * x + params.tau * y,
            sac.feature_extractor_target,
            new_fe,
        )
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
            new_lambda1_critic_target = jax.tree.map(
                lambda x, y: (1 - params.tau) * x + params.tau * y,
                sac.lambda1_critic_target,
                new_lambda1_critic,
            )
            new_lambda2_critic_target = jax.tree.map(
                lambda x, y: (1 - params.tau) * x + params.tau * y,
                sac.lambda2_critic_target,
                new_lambda2_critic,
            )
        return (
            SACState(
                new_fe,
                new_actor,  # type: ignore
                new_critic,  # type: ignore
                new_lambda1_critic if approximate_lambda else sac.lambda1_critic,
                new_lambda2_critic if approximate_lambda else sac.lambda2_critic,
                new_fe_target,
                new_actor_target,
                new_critic_target,
                (
                    new_lambda1_critic_target
                    if approximate_lambda
                    else sac.lambda1_critic_target
                ),
                (
                    new_lambda2_critic_target
                    if approximate_lambda
                    else sac.lambda2_critic_target
                ),
                new_fe_opt,
                new_actor_opt,
                new_critic_opt,
                (
                    new_lambda1_critic_opt
                    if approximate_lambda
                    else sac.lambda1_critic_optimizer_state
                ),
                (
                    new_lambda2_critic_opt
                    if approximate_lambda
                    else sac.lambda2_critic_optimizer_state
                ),
                new_alpha_opt,  # type: ignore
                new_alpha,
                new_log_alpha,  # type: ignore
                sac.online_buffer,
            ),
            metrics,
        )

    # Single-row (unbatched) zero carry for the env-rollout policy queries.
    _env_zero_carry = feature_extractor.initialize_carry(1)[0]

    def _train_unrolled(
        sac: SACState,
        env,
        env_params,
        env_state,
        env_carry,
        key: jax.Array,
    ) -> Tuple[SACState, any, any, dict]:
        """Pure ``train_steps``-long (env-step + grad-update) scan.

        No host-side control flow and no jit, so it is safe to ``jax.vmap``
        over a leading seed axis (stack ``sac`` / ``env_state`` / ``env_carry``
        / ``key``, keep ``env`` / ``env_params`` broadcast) to train many seeds
        concurrently in one kernel.  The caller is responsible for pre-filling
        the buffer first (see :func:`prefill_buffer`); :func:`train` wraps this
        with the automatic warm-up check, which is itself not vmap-safe.
        """

        def scan_fun(carry, _):
            sac, env_state, env_carry, key = carry
            key, next_key, env_key, update_key = jax.random.split(key, 4)
            sac, env_state, env_carry = run_env_step(
                sac, env, env_params, env_state, env_carry, env_key
            )
            sac, metrics = update_step(sac, update_key)
            return (sac, env_state, env_carry, next_key), metrics

        (sac, env_state, env_carry, _), metrics = jax.lax.scan(
            scan_fun, (sac, env_state, env_carry, key), length=train_steps
        )
        metrics = jax.tree.map(lambda x: x.mean(), metrics)
        return sac, env_state, env_carry, metrics

    @partial(jax.jit, static_argnames=["env"])
    def _train_jit(
        sac: SACState,
        env,
        env_params,
        env_state,
        env_carry,
        key: jax.Array,
    ) -> Tuple[SACState, any, any, dict]:
        print("compiling...")
        return _train_unrolled(sac, env, env_params, env_state, env_carry, key)

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
        4. Runs one joint gradient update via :func:`update_step`.

        The entire loop is compiled as a single XLA program after the first
        invocation (via the ``_train_jit`` inner function).

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
                sac, env, env_params, env_state, params.online_batch_size*(params.lambda_truncation+params.sequence_length+params.burn_in_length), prefill_key
            )
        # Start each train() call from a fresh zero carry.  Inside the scan the
        # carry is reset again on every episode boundary, so the only state lost
        # here is in-flight memory of an ongoing episode — acceptable until we
        # plumb carry persistence across train() invocations.
        sac, env_state, _new_carry, metrics = _train_jit(
            sac, env, env_params, env_state, _env_zero_carry, key
        )
        return sac, env_state, metrics

    @partial(jax.jit, static_argnames=["env", "n_steps"])
    def _prefill_jit(sac, env, env_params, env_state, n_steps, key):
        print("compiling prefill...")

        def scan_body(carry, step_idx):
            sac, env_state, key = carry
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
            terminated = done | (step_idx == n_steps - 1)
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
            return (sac, env_state, key), None

        (sac, env_state, _), _ = jax.lax.scan(
            scan_body, (sac, env_state, key), jnp.arange(n_steps)
        )
        return sac, env_state

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
        sampleable by :func:`update_step`.

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
        return _prefill_jit(sac, env, env_params, env_state, n_steps, key)

    fns = SACFunctions(
        predict, train, get_importance_ratios, prefill_buffer, _train_unrolled
    )
    if debug:
        return (
            iqlearn,
            fns,
            DebugFunctions(get_q, get_entropy),
        )
    return (iqlearn, fns)
