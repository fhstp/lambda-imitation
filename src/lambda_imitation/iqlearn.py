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
from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.tree_util import Partial

from .buffer import (Buffer, BufferFunctions, BufferSample, SequenceSample,
                     create_buffer, create_sample, create_sequence_sample)

# Bounds for the squashed log-standard-deviation of the policy distribution.
# The raw output is tanh-squashed and then rescaled into this range to keep
# the distribution numerically stable while remaining expressive.
LOG_STD_MIN = -5
LOG_STD_MAX = 2


def _h(x: jax.Array, eps: float) -> jax.Array:
    """Invertible value rescaling h (Pohlen 2018 / R2D2 §2.3)."""
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1.0) - 1.0) + eps * x


def _h_inv(z: jax.Array, eps: float) -> jax.Array:
    """Inverse of _h."""
    n = jnp.sqrt(4.0 * eps * (jnp.abs(z) + 1.0 + eps) + 1.0) - 1.0
    return jnp.sign(z) * ((n / (2.0 * eps)) ** 2 - 1.0)


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
        self.layers = nnx.data([
            nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)
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


class IdentityFeatureExtractor(nnx.Module):
    """No-op feature extractor — flattens the observation, no parameters.

    Structurally interchangeable with :class:`MLPFeatureExtractor` so the
    factory can treat all FE slots uniformly.  The output is the input with
    every non-batch dimension flattened, so the downstream feature dim equals
    the flat observation size.

    Use this (or pass ``None`` for the corresponding ``*_feature_extractor``
    argument to :func:`create_iqlearn`) when the head should consume raw
    observations directly without any learned encoder in front of it.

    Args:
        rngs: Accepted for API compatibility; not used (no parameters).
    """

    def __init__(self, *, rngs: nnx.Rngs | None = None):
        del rngs

    def __call__(self, x: jax.Array) -> jax.Array:
        return x.reshape(x.shape[0], -1)


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
        self.layers = nnx.data([
            nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)
        ])

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


class IdentityMemory(nnx.Module):
    """No-op memory module — passes features through unchanged.

    Structurally identical to ``LSTMMemory`` so the factory can treat all
    memory slots uniformly.  The carry is a zero-length array that costs
    nothing to carry through optimisers and env-step loops.

    Use this (or ``None`` in ``create_iqlearn``) to keep purely feedforward
    behaviour while preserving the three-module (FE → memory → head) structure.

    Args:
        feature_dim: Dimensionality of the incoming feature vector.  Stored
            so ``initial_carry`` can produce a correctly-shaped dummy carry.
        rngs: Accepted for API compatibility; not used (no parameters).
    """

    def __init__(self, feature_dim: int, *, rngs: nnx.Rngs):
        self.feature_dim = feature_dim

    def __call__(self, x: jax.Array, carry: jax.Array) -> tuple[jax.Array, jax.Array]:
        return carry, x

    def scan(
        self,
        xs: jax.Array,
        carry: jax.Array,
        dones: jax.Array | None = None,
        return_input_carries: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Run over a sequence — identity, so carry and output are unchanged.

        Args:
            xs: Feature sequence of shape ``(batch, T, feature_dim)``.
            carry: Current carry ``(batch, 0)``.
            dones: Ignored — zero-length carry has no state to reset.
            return_input_carries: If True, also return per-step input carries
                of shape ``(batch, T, *carry)`` (broadcast of the zero carry).

        Returns:
            ``(carry, xs)`` — carry unchanged, output equals input.  When
            ``return_input_carries=True``, a third element with the per-step
            input carries is appended.
        """
        del dones
        if return_input_carries:
            T = xs.shape[1]
            input_carries = jnp.broadcast_to(
                carry[:, None, :], (carry.shape[0], T, carry.shape[1])
            )
            return carry, xs, input_carries
        return carry, xs

    def initial_carry(self, batch_size: int) -> jax.Array:
        return jnp.zeros((batch_size, 0))


class LSTMMemory(nnx.Module):
    """LSTM recurrent memory — wraps ``nnx.LSTMCell``.

    Inserts an LSTM between the feature extractor and the head.  The carry is
    an ``LSTMState(c, h)`` NamedTuple where each component has shape
    ``(batch, hidden_dim)``.

    The caller is responsible for threading the carry between timesteps during
    environment interaction (via ``IQLearnState.actor_online_carry``) and
    during sequence training (via explicit carry arguments).

    Args:
        feature_dim: Dimensionality of the incoming feature vector (i.e. the
            output of the feature extractor).
        hidden_dim: Number of LSTM hidden units.
        rngs: Flax NNX RNG container used to initialise cell parameters.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.cell = nnx.LSTMCell(feature_dim, hidden_dim, rngs=rngs)
        self.hidden_dim = hidden_dim

    def __call__(self, x: jax.Array, carry) -> tuple[any, jax.Array]:
        """Run one LSTM step.

        Args:
            x: Feature batch of shape ``(batch, feature_dim)``.
            carry: ``LSTMState(c, h)`` from the previous step, each component
                of shape ``(batch, hidden_dim)``.

        Returns:
            ``(new_carry, y)`` where ``y`` has shape ``(batch, hidden_dim)``.
        """
        carry = (carry[:, : self.hidden_dim], carry[:, self.hidden_dim :])
        new_carry, y = self.cell(carry, x)
        new_carry = jnp.concatenate(new_carry, axis=1)
        return new_carry, y

    def scan(
        self,
        xs: jax.Array,
        initial_carry,
        dones: jax.Array | None = None,
        return_input_carries: bool = False,
    ) -> tuple[any, jax.Array]:
        """Run the LSTM over a full sequence via ``jax.lax.scan``.

        Args:
            xs: Feature sequence of shape ``(batch, T, feature_dim)``.
            initial_carry: ``LSTMState(c, h)`` each of shape ``(batch, hidden_dim)``.
            dones: Optional float32 terminal mask of shape ``(batch, T)``.  When
                ``dones[b, t] == 1``, the carry *leaving* step ``t`` is reset to
                zero before being fed into step ``t + 1``, so a terminal step
                still sees its pre-terminal carry but the next episode starts
                fresh.
            return_input_carries: If True, also return per-step input carries
                of shape ``(batch, T, 2*hidden_dim)`` — i.e. the carry that
                fed step ``t``.  Step 0's input carry equals ``initial_carry``.

        Returns:
            ``(final_carry, ys)`` where ``ys`` has shape ``(batch, T, hidden_dim)``.
            When ``return_input_carries=True``, a third element with the
            per-step input carries is appended.
        """

        xs_T = xs.swapaxes(0, 1)  # (T, batch, feature_dim)
        if dones is None:
            dones_T = jnp.zeros(xs_T.shape[:2], dtype=jnp.float32)
        else:
            dones_T = dones.swapaxes(0, 1)

        if return_input_carries:
            def step_with_input(carry, x_and_done):
                x_t, done_t = x_and_done
                input_carry = carry
                carry_split = (carry[:, : self.hidden_dim], carry[:, self.hidden_dim :])
                new_carry, y_t = self.cell(carry_split, x_t)
                new_carry = jnp.concatenate(new_carry, axis=1)
                new_carry = jnp.where(done_t[:, None], jnp.zeros_like(new_carry), new_carry)
                return new_carry, (y_t, input_carry)

            final_carry, (ys_T, input_carries_T) = jax.lax.scan(
                step_with_input, initial_carry, (xs_T, dones_T)
            )
            return (
                final_carry,
                ys_T.swapaxes(0, 1),
                input_carries_T.swapaxes(0, 1),
            )

        def step(carry, x_and_done):
            x_t, done_t = x_and_done
            carry = (carry[:, : self.hidden_dim], carry[:, self.hidden_dim :])
            new_carry, y_t = self.cell(carry, x_t)
            new_carry = jnp.concatenate(new_carry, axis=1)
            new_carry = jnp.where(done_t[:, None], jnp.zeros_like(new_carry), new_carry)
            return new_carry, y_t

        final_carry, ys_T = jax.lax.scan(step, initial_carry, (xs_T, dones_T))
        return final_carry, ys_T.swapaxes(0, 1)  # (batch, T, hidden_dim)

    def initial_carry(self, batch_size: int):
        """Return zero-initialised (c, h) each of shape ``(batch_size, hidden_dim)``."""
        return jnp.zeros((batch_size, self.hidden_dim * 2))


# ---------------------------------------------------------------------------
# State / function / graph containers
# ---------------------------------------------------------------------------


class NetworkState(NamedTuple):
    """Flax NNX graph states for a feature-extractor + memory + head triple.

    All fields are ``nnx.GraphState`` objects produced by ``nnx.split``.
    Together they form a JAX pytree, so optimizer updates and EMA target
    updates work on them transparently via ``jax.tree.map``.

    Attributes:
        fe: Graph state of the feature extractor module.
        memory: Graph state of the recurrent memory module (empty for
            :class:`IdentityMemory`).
        head: Graph state of the task-specific head module.
    """

    fe: nnx.GraphState
    memory: nnx.GraphState
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
    """Flax NNX graph definitions for a feature-extractor + memory + head triple.

    These are the static (non-parameter) graph descriptions produced by
    ``nnx.split`` and consumed by ``nnx.merge`` to reconstruct live modules
    during forward passes.

    Attributes:
        fe: Graph definition of the feature extractor.
        memory: Graph definition of the recurrent memory module.
        head: Graph definition of the head.
    """

    fe: nnx.GraphDef
    memory: nnx.GraphDef
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
    mc_critic: TwinCriticState
    actor_target: NetworkState
    critic_target: TwinCriticState
    mc_critic_target: TwinCriticState
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    mc_critic_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    alpha: jax.Array
    log_alpha: jax.Array
    online_buffer: Buffer
    actor_online_carry: any
    critic_q1_online_carry: any
    critic_q2_online_carry: any
    mc_critic_q1_online_carry: any
    mc_critic_q2_online_carry: any


class IQLearnFunctions(NamedTuple):
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
    train_sac: Callable
    get_importance_ratios: Callable
    prefill_buffer: Callable


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


class DebugFunctions(NamedTuple):
    """Optional debug helpers returned by :func:`create_iqlearn` when ``debug=True``.

    These functions expose internal computations that are useful for
    introspection and unit testing but are not required for normal training.

    Attributes:
        calculate_td_lambda: ``(actor, mc_critic_target, online_buffer, indices) ->
            td_returns`` -- compute a V-trace TD(λ) return estimate for each
            index in ``indices`` (vmapped).  ``actor`` is the current
            :class:`NetworkState` from :class:`IQLearnState`.
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
    run_actor_scan: Callable
    run_critic_scan: Callable


class SequenceSample(NamedTuple):
    """Pre-sampled sequence + burned-in carries for all networks.

    Produced once per update step by ``sample_with_burn_in`` and shared
    across all loss functions, avoiding redundant buffer reads and
    duplicate actor/critic burn-in passes.

    ``burn_ac_tgt``, ``burn_mc_*`` fields are ``None`` when the
    corresponding networks were not passed to ``sample_with_burn_in``.
    """

    obs: Any
    act: Any
    rew: Any
    done: Any
    mask: Any
    seq_idx: Any
    burn_ac: Any        # actor carry
    burn_ac_tgt: Any    # actor_target carry (None if not provided)
    burn_cq1: Any       # online critic q1 carry
    burn_cq2: Any       # online critic q2 carry
    burn_cq1_tgt: Any   # critic_target q1 carry
    burn_cq2_tgt: Any   # critic_target q2 carry
    burn_mc_cq1: Any    # mc_critic q1 carry (None if not provided)
    burn_mc_cq2: Any    # mc_critic q2 carry
    burn_mc_cq1_tgt: Any  # mc_critic_target q1 carry (None if not provided)
    burn_mc_cq2_tgt: Any  # mc_critic_target q2 carry


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
    mc_critic_lr: float = 1e-2
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
    lam: float = 0.5  # lambda is a reserved keyword...
    lambda_truncation: int = 15
    # R2D2 sequence replay (sequence_length=1 keeps the IID code path)
    sequence_length: int = 1
    burn_in_length: int = 0
    # If True, initialise burn-in (or the sequence scan when burn_in_length=0)
    # from the carry stored in the buffer at the start of the sampled window
    # instead of zero.  Requires sequence_length > 1 or burn_in_length > 0.
    burn_in_from_stored_carry: bool = False
    # n-step bootstrapping (1 = standard 1-step TD)
    n_step: int = 1
    # Invertible value rescaling (Pohlen 2018 / R2D2 §2.3)
    value_rescaling: bool = False
    value_rescaling_eps: float = 1e-3
    # Lambda-discrepancy auxiliary loss: Huber penalty on
    # (Q_sac + α·H_π) − Q_mc, applied to actor, SAC critic, and MC critic with
    # stop_gradient on every network except the one being optimised.
    # 0 disables the term entirely; non-zero requires approximate_mc=True.
    lambda_discrepancy_coef: float = 0.0
    lambda_discrepancy_delta: float = 1.0
    # Carry refresh: after each gradient step, write the per-step input
    # carries that the (pre-step) online networks just produced back into
    # the online buffer at the sampled slots seq_idx[:, B:B+SL].
    # Counters carry drift — without it, slots that were sampled long ago
    # carry hidden state from a stale earlier policy.  Requires
    # burn_in_from_stored_carry=True (otherwise stored carries are unread).
    refresh_stored_carries: bool = False
    # Global-norm gradient clipping applied to actor, critic, and (when active)
    # mc_critic optimisers via optax.clip_by_global_norm.  0.0 disables clipping.
    grad_clip_norm: float = 0.0
    # Per-step diagnostic metrics (grad/update/param global norms, max-abs grads,
    # carry norms).  Each walks the full pytree — non-trivial cost in the hot
    # update loop.  Disable for max throughput.
    diagnostics: bool = True


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
actor_carry_key = "carry_actor"
critic_q1_carry_key = "carry_critic_q1"
critic_q2_carry_key = "carry_critic_q2"
mc_critic_q1_carry_key = "carry_mc_critic_q1"
mc_critic_q2_carry_key = "carry_mc_critic_q2"


def create_iqlearn(
    params: Hyperparameters,
    buffer: Buffer,
    action_dim: int,
    actor_feature_extractor: nnx.Module | None,
    critic_q1_feature_extractor: nnx.Module | None,
    critic_q2_feature_extractor: nnx.Module | None,
    create_key: jax.Array,
    mc_critic_q1_feature_extractor: nnx.Module | None = None,
    mc_critic_q2_feature_extractor: nnx.Module | None = None,
    actor_memory: nnx.Module | None = None,
    critic_q1_memory: nnx.Module | None = None,
    critic_q2_memory: nnx.Module | None = None,
    mc_critic_q1_memory: nnx.Module | None = None,
    mc_critic_q2_memory: nnx.Module | None = None,
    obs_key: str = "observations",
    action_key: str = "actions",
    action_scale: float | jax.Array = 1,
    action_bias: float | jax.Array = 0,
    train_steps: int = 1000,
    actor_head_dims: tuple[int, ...] = (),
    critic_head_dims: tuple[int, ...] = (256, 256),
    mc_critic_head_dims: tuple[int, ...] = (256, 256),
    is_discrete: bool = False,
    approximate_mc: bool = False,
    debug: bool = False,
) -> (
    "Tuple[IQLearnState, IQLearnFunctions, IQLearnGraphs] | "
    "Tuple[IQLearnState, IQLearnFunctions, IQLearnGraphs, DebugFunctions]"
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
            afterwards.  Pass ``None`` to use a parameter-free
            :class:`IdentityFeatureExtractor` that flattens observations and
            feeds them straight into the head.
        critic_q1_feature_extractor: Same contract as ``actor_feature_extractor``.
            Used exclusively by the first Q-branch.  ``None`` synthesises an
            :class:`IdentityFeatureExtractor`.
        critic_q2_feature_extractor: Same contract as ``actor_feature_extractor``.
            Used exclusively by the second Q-branch.  Should be initialised with
            a different seed from ``critic_q1_feature_extractor`` to ensure the
            two branches start with different weights.  ``None`` synthesises an
            :class:`IdentityFeatureExtractor` (parameter-free, so identical
            across branches — branch divergence then comes from head init only).
        create_key: JAX PRNG key used to initialise all components that are
            created *inside* :func:`create_iqlearn` — the actor head, both
            critic heads, both MC-critic heads (when ``approximate_mc=True``),
            and any synthesised :class:`IdentityMemory` modules.  Sub-keys are
            derived from this key via :func:`jax.random.split`, so a single key
            fully determines the initial parameter distribution without any
            hidden global state.  Feature extractors passed in by the caller
            are *not* affected.
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
        ``(IQLearnState, IQLearnFunctions, IQLearnGraphs)`` triple.

        When ``debug=True``: a
        ``(IQLearnState, IQLearnFunctions, IQLearnGraphs, DebugFunctions)``
        4-tuple.

        - ``IQLearnState``: initial agent state with online and target networks
          set to the same weights.
        - ``IQLearnFunctions``: named tuple of ``predict`` and ``train`` closures.
        - ``IQLearnGraphs``: static NNX graph definitions for all network modules,
          useful for inspection or custom inference.
        - ``DebugFunctions`` *(only when debug=True)*: named tuple of internal
          helper functions, including ``calculate_td_lambda`` and ``get_q``.
    """
    if params.burn_in_from_stored_carry and (
        params.sequence_length <= 1 and params.burn_in_length <= 0
    ):
        raise ValueError(
            "burn_in_from_stored_carry requires sequence_length > 1 or "
            "burn_in_length > 0 so carry keys are registered in the online "
            "buffer sequence sample."
        )
    if params.lambda_discrepancy_coef != 0.0 and not approximate_mc:
        raise ValueError(
            "lambda_discrepancy_coef != 0 requires approximate_mc=True so the "
            "MC critic exists."
        )
    if params.refresh_stored_carries and not params.burn_in_from_stored_carry:
        raise ValueError(
            "refresh_stored_carries requires burn_in_from_stored_carry=True; "
            "otherwise the stored carries are never read and the refresh is "
            "a dead write."
        )
    this_keys = [obs_key, action_key]
    next_keys = [obs_key]
    buffer_sample = create_sample(
        buffer.size,
        params.batch_size,
        this_keys=this_keys,
        next_keys=next_keys,
    )

    # Derive per-component sub-keys from create_key so that all internally
    # created modules (heads and synthesised memories) are fully determined
    # by the single key supplied by the caller.
    # Indices 0-4: heads (actor, cq1, cq2, mc_cq1, mc_cq2)
    # Indices 5-9: synthesised memories (actor, cq1, cq2, mc_cq1, mc_cq2)
    _sub_keys = jax.random.split(create_key, 10)

    def _rngs(i: int) -> nnx.Rngs:
        """Return an nnx.Rngs seeded from the i-th sub-key."""
        return nnx.Rngs(int(jax.random.bits(_sub_keys[i])))

    # Synthesise IdentityFeatureExtractor when caller passes None — flattens
    # observations, no parameters, output dim equals the flat obs size.
    if actor_feature_extractor is None:
        actor_feature_extractor = IdentityFeatureExtractor()
    if critic_q1_feature_extractor is None:
        critic_q1_feature_extractor = IdentityFeatureExtractor()
    if critic_q2_feature_extractor is None:
        critic_q2_feature_extractor = IdentityFeatureExtractor()
    if approximate_mc:
        if mc_critic_q1_feature_extractor is None:
            mc_critic_q1_feature_extractor = IdentityFeatureExtractor()
        if mc_critic_q2_feature_extractor is None:
            mc_critic_q2_feature_extractor = IdentityFeatureExtractor()

    # Infer FE output dims via dummy forward pass (before split)
    dummy_obs = jnp.zeros((1,) + buffer.info[obs_key].shape[1:])
    actor_feature_dim = actor_feature_extractor(dummy_obs).shape[-1]
    critic_q1_feature_dim = critic_q1_feature_extractor(dummy_obs).shape[-1]
    critic_q2_feature_dim = critic_q2_feature_extractor(dummy_obs).shape[-1]
    if approximate_mc:
        mc_critic_q1_feature_dim = mc_critic_q1_feature_extractor(dummy_obs).shape[-1]
        mc_critic_q2_feature_dim = mc_critic_q2_feature_extractor(dummy_obs).shape[-1]

    # Synthesise IdentityMemory when caller passes None
    if actor_memory is None:
        actor_memory = IdentityMemory(actor_feature_dim, rngs=_rngs(5))
    if critic_q1_memory is None:
        critic_q1_memory = IdentityMemory(critic_q1_feature_dim, rngs=_rngs(6))
    if critic_q2_memory is None:
        critic_q2_memory = IdentityMemory(critic_q2_feature_dim, rngs=_rngs(7))
    if approximate_mc:
        if mc_critic_q1_memory is None:
            mc_critic_q1_memory = IdentityMemory(
                mc_critic_q1_feature_dim, rngs=_rngs(8)
            )
        if mc_critic_q2_memory is None:
            mc_critic_q2_memory = IdentityMemory(
                mc_critic_q2_feature_dim, rngs=_rngs(9)
            )

    # Infer memory output dims via second dummy forward pass
    _dummy_feat_actor = actor_feature_extractor(dummy_obs)
    _dummy_carry_actor = actor_memory.initial_carry(1)
    _, _dummy_mem_out_actor = actor_memory(_dummy_feat_actor, _dummy_carry_actor)
    actor_memory_out_dim = _dummy_mem_out_actor.shape[-1]

    _dummy_feat_q1 = critic_q1_feature_extractor(dummy_obs)
    _dummy_carry_q1 = critic_q1_memory.initial_carry(1)
    _, _dummy_mem_out_q1 = critic_q1_memory(_dummy_feat_q1, _dummy_carry_q1)
    critic_q1_memory_out_dim = _dummy_mem_out_q1.shape[-1]

    _dummy_feat_q2 = critic_q2_feature_extractor(dummy_obs)
    _dummy_carry_q2 = critic_q2_memory.initial_carry(1)
    _, _dummy_mem_out_q2 = critic_q2_memory(_dummy_feat_q2, _dummy_carry_q2)
    critic_q2_memory_out_dim = _dummy_mem_out_q2.shape[-1]

    if approximate_mc:
        _dummy_feat_mc_q1 = mc_critic_q1_feature_extractor(dummy_obs)
        _dummy_carry_mc_q1 = mc_critic_q1_memory.initial_carry(1)
        _, _dummy_mem_out_mc_q1 = mc_critic_q1_memory(
            _dummy_feat_mc_q1, _dummy_carry_mc_q1
        )
        mc_critic_q1_memory_out_dim = _dummy_mem_out_mc_q1.shape[-1]

        _dummy_feat_mc_q2 = mc_critic_q2_feature_extractor(dummy_obs)
        _dummy_carry_mc_q2 = mc_critic_q2_memory.initial_carry(1)
        _, _dummy_mem_out_mc_q2 = mc_critic_q2_memory(
            _dummy_feat_mc_q2, _dummy_carry_mc_q2
        )
        mc_critic_q2_memory_out_dim = _dummy_mem_out_mc_q2.shape[-1]

    # Carry templates (shape (1, *carry_shape)) used to build zero carries at
    # any batch size without calling initial_carry again inside JIT.
    _actor_carry_template = actor_memory.initial_carry(1)
    _critic_q1_carry_template = critic_q1_memory.initial_carry(1)
    _critic_q2_carry_template = critic_q2_memory.initial_carry(1)
    if approximate_mc:
        _mc_critic_q1_carry_template = mc_critic_q1_memory.initial_carry(1)
        _mc_critic_q2_carry_template = mc_critic_q2_memory.initial_carry(1)

    def _make_zero_carry(template, batch_size: int):
        return jax.tree.map(lambda t: jnp.zeros((batch_size,) + t.shape[1:]), template)

    def _make_actor_carry(batch_size: int):
        return _make_zero_carry(_actor_carry_template, batch_size)

    def _make_critic_q1_carry(batch_size: int):
        return _make_zero_carry(_critic_q1_carry_template, batch_size)

    def _make_critic_q2_carry(batch_size: int):
        return _make_zero_carry(_critic_q2_carry_template, batch_size)

    if approximate_mc:

        def _make_mc_critic_q1_carry(batch_size: int):
            return _make_zero_carry(_mc_critic_q1_carry_template, batch_size)

        def _make_mc_critic_q2_carry(batch_size: int):
            return _make_zero_carry(_mc_critic_q2_carry_template, batch_size)

    # Online buffer: same obs/action shapes as the expert buffer, plus scalar
    # reward and terminated fields written by run_env_step / train_sac.
    online_shapes = {
        **extract_buffer_shapes(buffer),
        reward_key: (),
        return_key: (),
        terminated_key: (),
        behaviour_key: (),
        actor_carry_key: _actor_carry_template.shape[1:],
        critic_q1_carry_key: _critic_q1_carry_template.shape[1:],
        critic_q2_carry_key: _critic_q2_carry_template.shape[1:],
    }

    if not is_discrete:
        online_shapes[unsquashed_action_key] = online_shapes[action_key]
    if approximate_mc:
        online_shapes[mc_critic_q1_carry_key] = _mc_critic_q1_carry_template.shape[1:]
        online_shapes[mc_critic_q2_carry_key] = _mc_critic_q2_carry_template.shape[1:]

    # TODO: this is old, non LSTM code, replace in create_buffer
    online_this_keys = [obs_key, action_key, reward_key, terminated_key]
    online_next_keys = [obs_key]
    online_buffer, online_buffer_functions = create_buffer(
        online_shapes,
        params.online_buffer_size,
        params.online_batch_size,
        online_this_keys,
        online_next_keys,
    )
    # set last index in buffer to terminated for correct carry burn in
    online_buffer = online_buffer._replace(
        info={
            **online_buffer.info,
            terminated_key: online_buffer.info[terminated_key].at[-1].set(1),
        }
    )

    online_buffer_sample = online_buffer_functions.sample
    _seq_keys = [obs_key, action_key, reward_key, terminated_key]
    if approximate_mc:
        _seq_keys.append(behaviour_key)
    if params.sequence_length > 1 or params.burn_in_length > 0:
        _seq_keys.append(actor_carry_key)
        _seq_keys.append(critic_q1_carry_key)
        _seq_keys.append(critic_q2_carry_key)
        if approximate_mc:
            _seq_keys.append(mc_critic_q1_carry_key)
            _seq_keys.append(mc_critic_q2_carry_key)
    online_sequence_sample_fn = create_sequence_sample(
        params.online_buffer_size,
        params.online_batch_size,
        params.burn_in_length,
        params.sequence_length
        + 1,  # +1 to always have the next obs available for updates
        _seq_keys,
        terminal_key=terminated_key,
    )

    # burn-in helper
    def sample_with_burn_in(
        online_buffer,
        actor,
        critic,
        critic_target,
        key_sample,
        use_mc=False,
        mc_critic=None,
        mc_critic_target=None,
        actor_target=None,
    ) -> SequenceSample:
        """Sample a sequence and prime all recurrent carries in one pass.

        Burns in actor, critic, and critic_target always.  When
        ``mc_critic`` / ``mc_critic_target`` are provided they are burned in
        using mc carry keys.  When ``actor_target`` is provided a second actor
        burn-in is run, producing ``burn_ac_tgt`` for use by the critic loss.

        Returns a :class:`SequenceSample` with all sequence data and carries.
        """
        _B = params.burn_in_length
        _SL = params.sequence_length + 1
        _bs = params.online_batch_size

        seq, seq_idx = online_sequence_sample_fn(online_buffer, key_sample)
        obs = seq.info[obs_key][:, _B:]
        act = seq.info[action_key][:, _B:]
        rew = seq.info[reward_key][:, _B:].reshape(_bs, _SL)
        done = seq.info[terminated_key][:, _B:].reshape(_bs, _SL)
        mask = seq.mask[:, _B:]

        # Carry keys and factories for the primary critic pair.
        if use_mc:
            _q1_key = mc_critic_q1_carry_key
            _q2_key = mc_critic_q2_carry_key
            _make_q1 = _make_mc_critic_q1_carry
            _make_q2 = _make_mc_critic_q2_carry
        else:
            _q1_key = critic_q1_carry_key
            _q2_key = critic_q2_carry_key
            _make_q1 = _make_critic_q1_carry
            _make_q2 = _make_critic_q2_carry

        _has_mc = mc_critic is not None or mc_critic_target is not None

        if params.burn_in_from_stored_carry:
            _init_ac = seq.info[actor_carry_key][:, 0]
            _init_cq1 = seq.info[_q1_key][:, 0]
            _init_cq2 = seq.info[_q2_key][:, 0]
            if _has_mc:
                _init_mc_cq1 = seq.info[mc_critic_q1_carry_key][:, 0]
                _init_mc_cq2 = seq.info[mc_critic_q2_carry_key][:, 0]
        else:
            _init_ac = _make_actor_carry(_bs)
            _init_cq1 = _make_q1(_bs)
            _init_cq2 = _make_q2(_bs)
            if _has_mc:
                _init_mc_cq1 = _make_mc_critic_q1_carry(_bs)
                _init_mc_cq2 = _make_mc_critic_q2_carry(_bs)

        if not _has_mc:
            _init_mc_cq1 = None
            _init_mc_cq2 = None

        if _B > 0:
            _burn_obs = seq.info[obs_key][:, :_B]
            _burn_done = seq.info[terminated_key][:, :_B].astype(jnp.float32)
            _burn_ac, _ = jax.lax.stop_gradient(
                run_actor_scan(actor, _burn_obs, _init_ac, _burn_done)
            )
            if actor_target is not None:
                _burn_ac_tgt, _ = jax.lax.stop_gradient(
                    run_actor_scan(actor_target, _burn_obs, _init_ac, _burn_done)
                )
            else:
                _burn_ac_tgt = None
            if is_discrete:
                _burn_cq1, _burn_cq2, _ = jax.lax.stop_gradient(
                    run_critic_scan(
                        critic, _burn_obs, use_mc, _init_cq1, _init_cq2, _burn_done
                    )
                )
                _burn_cq1_tgt, _burn_cq2_tgt, _ = jax.lax.stop_gradient(
                    run_critic_scan(
                        critic_target,
                        _burn_obs,
                        use_mc,
                        _init_cq1,
                        _init_cq2,
                        _burn_done,
                    )
                )
                if mc_critic is not None:
                    _burn_mc_cq1, _burn_mc_cq2, _ = jax.lax.stop_gradient(
                        run_critic_scan(
                            mc_critic,
                            _burn_obs,
                            True,
                            _init_mc_cq1,
                            _init_mc_cq2,
                            _burn_done,
                        )
                    )
                else:
                    _burn_mc_cq1, _burn_mc_cq2 = _init_mc_cq1, _init_mc_cq2
                if mc_critic_target is not None:
                    _burn_mc_cq1_tgt, _burn_mc_cq2_tgt, _ = jax.lax.stop_gradient(
                        run_critic_scan(
                            mc_critic_target,
                            _burn_obs,
                            True,
                            _init_mc_cq1,
                            _init_mc_cq2,
                            _burn_done,
                        )
                    )
                else:
                    _burn_mc_cq1_tgt, _burn_mc_cq2_tgt = _init_mc_cq1, _init_mc_cq2
            else:
                _burn_act = seq.info[action_key][:, :_B]
                _burn_cq1, _burn_cq2, _ = jax.lax.stop_gradient(
                    run_critic_scan(
                        critic,
                        _burn_obs,
                        _burn_act,
                        use_mc,
                        _init_cq1,
                        _init_cq2,
                        _burn_done,
                    )
                )
                _burn_cq1_tgt, _burn_cq2_tgt, _ = jax.lax.stop_gradient(
                    run_critic_scan(
                        critic_target,
                        _burn_obs,
                        _burn_act,
                        use_mc,
                        _init_cq1,
                        _init_cq2,
                        _burn_done,
                    )
                )
                if mc_critic is not None:
                    _burn_mc_cq1, _burn_mc_cq2, _ = jax.lax.stop_gradient(
                        run_critic_scan(
                            mc_critic,
                            _burn_obs,
                            _burn_act,
                            True,
                            _init_mc_cq1,
                            _init_mc_cq2,
                            _burn_done,
                        )
                    )
                else:
                    _burn_mc_cq1, _burn_mc_cq2 = _init_mc_cq1, _init_mc_cq2
                if mc_critic_target is not None:
                    _burn_mc_cq1_tgt, _burn_mc_cq2_tgt, _ = jax.lax.stop_gradient(
                        run_critic_scan(
                            mc_critic_target,
                            _burn_obs,
                            _burn_act,
                            True,
                            _init_mc_cq1,
                            _init_mc_cq2,
                            _burn_done,
                        )
                    )
                else:
                    _burn_mc_cq1_tgt, _burn_mc_cq2_tgt = _init_mc_cq1, _init_mc_cq2
        else:
            _burn_ac = _init_ac
            _burn_ac_tgt = None if actor_target is None else _init_ac
            _burn_cq1 = _init_cq1
            _burn_cq2 = _init_cq2
            _burn_cq1_tgt = _init_cq1
            _burn_cq2_tgt = _init_cq2
            _burn_mc_cq1 = _init_mc_cq1
            _burn_mc_cq2 = _init_mc_cq2
            _burn_mc_cq1_tgt = _init_mc_cq1
            _burn_mc_cq2_tgt = _init_mc_cq2
        return SequenceSample(
            obs=obs,
            act=act,
            rew=rew,
            done=done,
            mask=mask,
            seq_idx=seq_idx,
            burn_ac=_burn_ac,
            burn_ac_tgt=_burn_ac_tgt,
            burn_cq1=_burn_cq1,
            burn_cq2=_burn_cq2,
            burn_cq1_tgt=_burn_cq1_tgt,
            burn_cq2_tgt=_burn_cq2_tgt,
            burn_mc_cq1=_burn_mc_cq1,
            burn_mc_cq2=_burn_mc_cq2,
            burn_mc_cq1_tgt=_burn_mc_cq1_tgt,
            burn_mc_cq2_tgt=_burn_mc_cq2_tgt,
        )

    if approximate_mc:
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

    # Create heads — discrete and continuous differ only in output_dim and
    # whether actions are concatenated to features before the head.
    # Sub-keys 0-4 are reserved for heads: 0=actor, 1=cq1, 2=cq2, 3=mc_cq1, 4=mc_cq2.
    if is_discrete:
        actor_head_model = Head(
            actor_memory_out_dim,
            actor_head_dims,
            action_dim,
            rngs=_rngs(0),
        )
        critic_q1_head_model = Head(
            critic_q1_memory_out_dim,
            critic_head_dims,
            action_dim,
            rngs=_rngs(1),
        )
        critic_q2_head_model = Head(
            critic_q2_memory_out_dim,
            critic_head_dims,
            action_dim,
            rngs=_rngs(2),
        )
        if approximate_mc:
            mc_critic_q1_head_model = Head(
                mc_critic_q1_memory_out_dim,
                mc_critic_head_dims,
                action_dim,
                rngs=_rngs(3),
            )
            mc_critic_q2_head_model = Head(
                mc_critic_q2_memory_out_dim,
                mc_critic_head_dims,
                action_dim,
                rngs=_rngs(4),
            )
    else:
        actor_head_model = Head(
            actor_memory_out_dim,
            actor_head_dims,
            2 * action_dim,
            rngs=_rngs(0),
        )
        # For continuous critics, features and actions are concatenated before
        # the head, so input_dim = memory_out_dim + action_dim, output_dim = 1.
        critic_q1_head_model = Head(
            critic_q1_memory_out_dim + action_dim,
            critic_head_dims,
            1,
            rngs=_rngs(1),
        )
        critic_q2_head_model = Head(
            critic_q2_memory_out_dim + action_dim,
            critic_head_dims,
            1,
            rngs=_rngs(2),
        )
        if approximate_mc:
            mc_critic_q1_head_model = Head(
                mc_critic_q1_memory_out_dim + action_dim,
                mc_critic_head_dims,
                1,
                rngs=_rngs(3),
            )
            mc_critic_q2_head_model = Head(
                mc_critic_q2_memory_out_dim + action_dim,
                mc_critic_head_dims,
                1,
                rngs=_rngs(4),
            )

    # Split all modules into (graph_def, state) — FE, memory, and head per network.
    # Memory modules are split into Params and the *rest* (rng counters, keys)
    # so that the trainable state is float-only.  The non-Param remainder is
    # captured in the closure and re-applied on every nnx.merge call; jax.grad
    # therefore never sees uint32 / PRNG-key leaves and can differentiate the
    # actor / critic states directly.
    actor_fe_graph, actor_fe_st = nnx.split(actor_feature_extractor)
    actor_memory_graph, actor_memory_st, actor_memory_rngs = nnx.split(
        actor_memory, nnx.Param, ...
    )
    actor_head_graph, actor_head_st = nnx.split(actor_head_model)
    critic_q1_fe_graph, critic_q1_fe_st = nnx.split(critic_q1_feature_extractor)
    critic_q1_memory_graph, critic_q1_memory_st, critic_q1_memory_rngs = nnx.split(
        critic_q1_memory, nnx.Param, ...
    )
    critic_q1_head_graph, critic_q1_head_st = nnx.split(critic_q1_head_model)
    critic_q2_fe_graph, critic_q2_fe_st = nnx.split(critic_q2_feature_extractor)
    critic_q2_memory_graph, critic_q2_memory_st, critic_q2_memory_rngs = nnx.split(
        critic_q2_memory, nnx.Param, ...
    )
    critic_q2_head_graph, critic_q2_head_st = nnx.split(critic_q2_head_model)
    if approximate_mc:
        mc_critic_q1_fe_graph, mc_critic_q1_fe_st = nnx.split(
            mc_critic_q1_feature_extractor
        )
        (
            mc_critic_q1_memory_graph,
            mc_critic_q1_memory_st,
            mc_critic_q1_memory_rngs,
        ) = nnx.split(mc_critic_q1_memory, nnx.Param, ...)
        mc_critic_q1_head_graph, mc_critic_q1_head_st = nnx.split(
            mc_critic_q1_head_model
        )
        mc_critic_q2_fe_graph, mc_critic_q2_fe_st = nnx.split(
            mc_critic_q2_feature_extractor
        )
        (
            mc_critic_q2_memory_graph,
            mc_critic_q2_memory_st,
            mc_critic_q2_memory_rngs,
        ) = nnx.split(mc_critic_q2_memory, nnx.Param, ...)
        mc_critic_q2_head_graph, mc_critic_q2_head_st = nnx.split(
            mc_critic_q2_head_model
        )

    actor_state = NetworkState(actor_fe_st, actor_memory_st, actor_head_st)
    critic_q1_state = NetworkState(
        critic_q1_fe_st, critic_q1_memory_st, critic_q1_head_st
    )
    critic_q2_state = NetworkState(
        critic_q2_fe_st, critic_q2_memory_st, critic_q2_head_st
    )
    critic_state = TwinCriticState(critic_q1_state, critic_q2_state)
    if approximate_mc:
        mc_critic_q1_state = NetworkState(
            mc_critic_q1_fe_st, mc_critic_q1_memory_st, mc_critic_q1_head_st
        )
        mc_critic_q2_state = NetworkState(
            mc_critic_q2_fe_st, mc_critic_q2_memory_st, mc_critic_q2_head_st
        )
        mc_critic_state = TwinCriticState(mc_critic_q1_state, mc_critic_q2_state)

    # Optimizers: actor operates on NetworkState; critic on TwinCriticState.
    def _make_optimizer(lr: float) -> optax.GradientTransformation:
        if params.grad_clip_norm > 0.0:
            return optax.chain(
                optax.clip_by_global_norm(params.grad_clip_norm),
                optax.adam(lr),
            )
        return optax.adam(lr)

    actor_optimizer = _make_optimizer(params.actor_lr)
    critic_optimizer = _make_optimizer(params.critic_lr)
    if approximate_mc:
        mc_critic_optimizer = _make_optimizer(params.mc_critic_lr)
    alpha_optimizer = optax.adam(params.alpha_lr)

    log_alpha = jnp.array(jnp.log(params.alpha))
    actor_optimizer_state = actor_optimizer.init(actor_state)
    critic_optimizer_state = critic_optimizer.init(critic_state)
    if approximate_mc:
        mc_critic_optimizer_state = mc_critic_optimizer.init(mc_critic_state)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    def remove_weak_types(state):
        """Materialise JAX weak dtypes to concrete dtypes to avoid JIT issues."""
        return jax.tree.map(
            lambda x: jnp.array(x, dtype=x.dtype) if hasattr(x, "dtype") else x,
            state,
        )

    # Initial online carries (batch_size=1 for single-step env interaction)
    actor_online_carry_init = remove_weak_types(actor_memory.initial_carry(1))
    critic_q1_online_carry_init = remove_weak_types(critic_q1_memory.initial_carry(1))
    critic_q2_online_carry_init = remove_weak_types(critic_q2_memory.initial_carry(1))
    if approximate_mc:
        mc_critic_q1_online_carry_init = remove_weak_types(
            mc_critic_q1_memory.initial_carry(1)
        )
        mc_critic_q2_online_carry_init = remove_weak_types(
            mc_critic_q2_memory.initial_carry(1)
        )
    else:
        mc_critic_q1_online_carry_init = None
        mc_critic_q2_online_carry_init = None

    iqlearn = IQLearnState(
        remove_weak_types(actor_state),
        remove_weak_types(critic_state),
        remove_weak_types(mc_critic_state) if approximate_mc else None,
        remove_weak_types(actor_state),  # targets start equal to online weights
        remove_weak_types(critic_state),
        remove_weak_types(mc_critic_state) if approximate_mc else None,
        remove_weak_types(actor_optimizer_state),
        remove_weak_types(critic_optimizer_state),
        remove_weak_types(mc_critic_optimizer_state) if approximate_mc else None,
        remove_weak_types(alpha_optimizer_state),
        remove_weak_types(jnp.exp(log_alpha)),
        remove_weak_types(log_alpha),
        remove_weak_types(online_buffer),
        actor_online_carry_init,
        critic_q1_online_carry_init,
        critic_q2_online_carry_init,
        mc_critic_q1_online_carry_init,
        mc_critic_q2_online_carry_init,
    )

    # ------------------------------------------------------------------
    # Shared helper: actor forward pass (same for both action space types)
    # ------------------------------------------------------------------

    def run_actor(actor: NetworkState, x: jax.Array, carry) -> tuple[any, jax.Array]:
        """Reconstruct and run the actor (FE → memory → head) on observation batch x."""
        fe = nnx.merge(actor_fe_graph, actor.fe)
        memory = nnx.merge(actor_memory_graph, actor.memory, actor_memory_rngs)
        head = nnx.merge(actor_head_graph, actor.head)
        new_carry, mem_out = memory(fe(x), carry)
        return new_carry, head(mem_out)

    def run_actor_scan(
        actor: NetworkState,
        xs: jax.Array,
        initial_carry,
        dones: jax.Array | None = None,
        return_input_carries: bool = False,
    ) -> tuple[any, jax.Array]:
        """Run the actor over a sequence of observations.

        Applies the feature extractor to all timesteps in one batched call,
        then threads the carry through the memory via ``memory.scan``, then
        applies the head to all memory outputs.

        Args:
            actor: Actor network state.
            xs: Observation sequence of shape ``(batch, T, *obs_shape)``.
            initial_carry: Initial memory carry from ``memory.initial_carry``.
            dones: Optional terminal mask ``(batch, T)``; the carry leaving a
                terminal step is zeroed before the next step.
            return_input_carries: If True, also return per-step input carries
                of shape ``(batch, T, *carry)`` — the carry fed *into* each
                step.  Step 0 receives ``initial_carry``.

        Returns:
            ``(final_carry, ys)`` where ``ys`` has shape
            ``(batch, T, output_dim)``.  When ``return_input_carries=True``,
            a third element with the per-step input carries is appended.
        """
        fe = nnx.merge(actor_fe_graph, actor.fe)
        memory = nnx.merge(actor_memory_graph, actor.memory, actor_memory_rngs)
        head = nnx.merge(actor_head_graph, actor.head)
        batch, T = xs.shape[0], xs.shape[1]
        flat_feats = fe(xs.reshape(batch * T, *xs.shape[2:]))
        feats = flat_feats.reshape(batch, T, -1)
        if return_input_carries:
            final_carry, mem_outs, input_carries = memory.scan(
                feats, initial_carry, dones, return_input_carries=True
            )
        else:
            final_carry, mem_outs = memory.scan(feats, initial_carry, dones)
        ys = head(mem_outs.reshape(batch * T, -1)).reshape(batch, T, -1)
        if return_input_carries:
            return final_carry, ys, input_carries
        return final_carry, ys

    # ------------------------------------------------------------------
    # Action-space-specific helpers
    # ------------------------------------------------------------------

    if is_discrete:

        def run_critic(
            critic: TwinCriticState,
            x: jax.Array,
            carry1: jax.Array,
            carry2: jax.Array,
            use_mc: bool = False,
        ) -> jax.Array:
            """Reconstruct and run both discrete critic branches.

            Returns:
                Array of shape ``(batch, num_actions, 2)`` where the last axis
                indexes the two independent Q estimates.
            """
            fe1 = nnx.merge(
                mc_critic_q1_fe_graph if use_mc else critic_q1_fe_graph, critic.q1.fe
            )
            mem1 = nnx.merge(
                mc_critic_q1_memory_graph if use_mc else critic_q1_memory_graph,
                critic.q1.memory,
                mc_critic_q1_memory_rngs if use_mc else critic_q1_memory_rngs,
            )
            head1 = nnx.merge(
                mc_critic_q1_head_graph if use_mc else critic_q1_head_graph,
                critic.q1.head,
            )
            fe2 = nnx.merge(
                mc_critic_q2_fe_graph if use_mc else critic_q2_fe_graph, critic.q2.fe
            )
            mem2 = nnx.merge(
                mc_critic_q2_memory_graph if use_mc else critic_q2_memory_graph,
                critic.q2.memory,
                mc_critic_q2_memory_rngs if use_mc else critic_q2_memory_rngs,
            )
            head2 = nnx.merge(
                mc_critic_q2_head_graph if use_mc else critic_q2_head_graph,
                critic.q2.head,
            )
            batch = x.shape[0]
            new_carry1, mem_out1 = mem1(fe1(x), carry1)
            new_carry2, mem_out2 = mem2(fe2(x), carry2)
            q1 = head1(mem_out1)  # (batch, num_actions)
            q2 = head2(mem_out2)  # (batch, num_actions)
            return (
                jnp.stack([q1, q2], axis=-1),
                new_carry1,
                new_carry2,
            )  # (batch, num_actions, 2)

        def run_critic_scan(
            critic: TwinCriticState,
            xs: jax.Array,
            use_mc: bool = False,
            initial_carry_q1=None,
            initial_carry_q2=None,
            dones: jax.Array | None = None,
            return_input_carries: bool = False,
        ) -> tuple[any, any, jax.Array]:
            """Run both discrete critic branches over an observation sequence.

            Args:
                critic: Twin-critic network state.
                xs: Observation sequence of shape ``(batch, T, *obs_shape)``.
                use_mc: If True, use MC critic graph definitions.
                initial_carry_q1: Initial carry for Q1 branch.  If ``None``,
                    a zero carry is created from the carry template.
                initial_carry_q2: Initial carry for Q2 branch.  If ``None``,
                    a zero carry is created from the carry template.
                dones: Optional terminal mask ``(batch, T)``; the carry leaving
                    a terminal step is zeroed before the next step.
                return_input_carries: If True, also return per-step input
                    carries for both branches.

            Returns:
                ``(final_carry_q1, final_carry_q2, qs)`` where ``qs`` has
                shape ``(batch, T, num_actions, 2)``.  When
                ``return_input_carries=True``, two additional elements
                ``(input_carries_q1, input_carries_q2)`` are appended.
            """
            fe1 = nnx.merge(
                mc_critic_q1_fe_graph if use_mc else critic_q1_fe_graph, critic.q1.fe
            )
            mem1 = nnx.merge(
                mc_critic_q1_memory_graph if use_mc else critic_q1_memory_graph,
                critic.q1.memory,
                mc_critic_q1_memory_rngs if use_mc else critic_q1_memory_rngs,
            )
            head1 = nnx.merge(
                mc_critic_q1_head_graph if use_mc else critic_q1_head_graph,
                critic.q1.head,
            )
            fe2 = nnx.merge(
                mc_critic_q2_fe_graph if use_mc else critic_q2_fe_graph, critic.q2.fe
            )
            mem2 = nnx.merge(
                mc_critic_q2_memory_graph if use_mc else critic_q2_memory_graph,
                critic.q2.memory,
                mc_critic_q2_memory_rngs if use_mc else critic_q2_memory_rngs,
            )
            head2 = nnx.merge(
                mc_critic_q2_head_graph if use_mc else critic_q2_head_graph,
                critic.q2.head,
            )
            batch, T = xs.shape[0], xs.shape[1]
            flat_xs = xs.reshape(batch * T, *xs.shape[2:])
            feats1 = fe1(flat_xs).reshape(batch, T, -1)
            feats2 = fe2(flat_xs).reshape(batch, T, -1)
            if initial_carry_q1 is None:
                initial_carry_q1 = (
                    _make_mc_critic_q1_carry(batch)
                    if use_mc
                    else _make_critic_q1_carry(batch)
                )
            if initial_carry_q2 is None:
                initial_carry_q2 = (
                    _make_mc_critic_q2_carry(batch)
                    if use_mc
                    else _make_critic_q2_carry(batch)
                )
            if return_input_carries:
                fc1, mem_outs1, in_c1 = mem1.scan(
                    feats1, initial_carry_q1, dones, return_input_carries=True
                )
                fc2, mem_outs2, in_c2 = mem2.scan(
                    feats2, initial_carry_q2, dones, return_input_carries=True
                )
            else:
                fc1, mem_outs1 = mem1.scan(feats1, initial_carry_q1, dones)
                fc2, mem_outs2 = mem2.scan(feats2, initial_carry_q2, dones)
            q1 = head1(mem_outs1.reshape(batch * T, -1)).reshape(batch, T, -1)
            q2 = head2(mem_outs2.reshape(batch * T, -1)).reshape(batch, T, -1)
            qs = jnp.stack([q1, q2], axis=-1)  # (batch, T, num_actions, 2)
            if return_input_carries:
                return fc1, fc2, qs, in_c1, in_c2
            return fc1, fc2, qs

        def get_q_both(
            critic: TwinCriticState,
            x: jax.Array,
            carry1: jax.Array,
            carry2: jax.Array,
            expert_actions: jax.Array,
            use_mc: bool = False,
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
            q_twin, new_carry1, new_carry2 = run_critic(
                critic, x, carry1, carry2, use_mc
            )  # (batch, num_actions, 2)
            indices = jnp.round(expert_actions.reshape(-1)).astype(jnp.int32)
            batch = q_twin.shape[0]
            return (
                q_twin[jnp.arange(batch), indices, 0],  # (batch,)
                q_twin[jnp.arange(batch), indices, 1],
                new_carry1,
                new_carry2,
            )  # (batch,)

        def get_q(
            critic: TwinCriticState,
            x: jax.Array,
            carry1: jax.Array,
            carry2: jax.Array,
            expert_actions: jax.Array,
            use_mc: bool = False,
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
            q1, q2, new_carry1, new_carry2 = get_q_both(
                critic, x, carry1, carry2, expert_actions, use_mc
            )
            return jnp.minimum(q1, q2), new_carry1, new_carry2

        def get_v(
            actor: NetworkState,
            critic: TwinCriticState,
            alpha: jax.Array,
            x: jax.Array,
            actor_carry: jax.Array,
            q_carry1: jax.Array,
            q_carry2: jax.Array,
            key: jax.Array,
            include_entropy: bool = True,
            include_log: bool = False,
            use_mc: bool = False,
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
            new_actor_carry, logits = run_actor(actor, x, actor_carry)
            probs = jax.nn.softmax(logits)  # (batch, num_actions)
            log_probs = jax.nn.log_softmax(logits)  # (batch, num_actions)
            q_twin, new_q_carry1, new_q_carry2 = run_critic(
                critic, x, q_carry1, q_carry2, use_mc
            )  # (batch, num_actions, 2)
            q_min = jnp.minimum(q_twin[..., 0], q_twin[..., 1])  # (batch, num_actions)
            entropy = -(probs * log_probs).sum(-1)  # (batch,) — exact H(π)
            if include_entropy:
                v = (probs * q_min).sum(-1) + alpha * entropy
                if include_log:
                    return (
                        v,
                        (new_actor_carry, new_q_carry1, new_q_carry2),
                        {
                            "q": (probs * q_min).sum(-1).mean(),
                            "entropy": entropy.mean(),
                        },
                    )
                return v, (new_actor_carry, new_q_carry1, new_q_carry2)
            else:
                return (probs * q_min).sum(-1), (
                    new_actor_carry,
                    new_q_carry1,
                    new_q_carry2,
                )

        def get_entropy(actor, obs, actor_carry, _key):
            _, logits = run_actor(actor, obs, actor_carry)
            probs = jax.nn.softmax(logits)  # (batch, num_actions)
            log_probs = jax.nn.log_softmax(logits)  # (batch, num_actions)
            entropy = -(probs * log_probs).sum(-1)  # (batch,) — exact H(π)
            return entropy

        @partial(jax.jit, static_argnames=["deterministic", "return_prob"])
        def predict(
            iqlearn: IQLearnState,
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
            new_carry, logits_batch = run_actor(iqlearn.actor, obs_batch, carry)
            logits = logits_batch[0]  # (num_actions,)
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
            return action.astype(jnp.float32), new_carry

        @jax.jit
        def get_importance_ratios(
            actor: NetworkState,
            obs: jax.Array,
            carry: jax.Array,
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
            _new_carry, logits = run_actor(actor, obs, carry)
            probs = jax.nn.softmax(logits)  # (batch, num_actions)
            idx = actions.reshape(-1).astype(jnp.int32)  # (batch,)
            pi_a = probs[jnp.arange(idx.shape[0]), idx]  # (batch,)
            return pi_a / behaviour_probs, _new_carry

    else:
        # ------------------------------------------------------------------
        # Continuous helpers
        # ------------------------------------------------------------------

        def run_critic(
            critic: TwinCriticState,
            x: jax.Array,
            actions: jax.Array,
            carry1: any = None,
            carry2: any = None,
            use_mc: bool = False,
        ) -> tuple[jax.Array, any, any]:
            """Reconstruct and run both continuous critic branches.

            Features pass through the memory then are concatenated with actions
            before each head so each branch has a fully independent view of the
            (obs, action) pair.

            Returns:
                ``(qs, new_carry1, new_carry2)`` where ``qs`` has shape
                ``(batch, 2)`` containing two independent Q estimates.
            """
            fe1 = nnx.merge(
                mc_critic_q1_fe_graph if use_mc else critic_q1_fe_graph, critic.q1.fe
            )
            mem1 = nnx.merge(
                mc_critic_q1_memory_graph if use_mc else critic_q1_memory_graph,
                critic.q1.memory,
                mc_critic_q1_memory_rngs if use_mc else critic_q1_memory_rngs,
            )
            head1 = nnx.merge(
                mc_critic_q1_head_graph if use_mc else critic_q1_head_graph,
                critic.q1.head,
            )
            fe2 = nnx.merge(
                mc_critic_q2_fe_graph if use_mc else critic_q2_fe_graph, critic.q2.fe
            )
            mem2 = nnx.merge(
                mc_critic_q2_memory_graph if use_mc else critic_q2_memory_graph,
                critic.q2.memory,
                mc_critic_q2_memory_rngs if use_mc else critic_q2_memory_rngs,
            )
            head2 = nnx.merge(
                mc_critic_q2_head_graph if use_mc else critic_q2_head_graph,
                critic.q2.head,
            )
            batch = x.shape[0]
            if carry1 is None:
                carry1 = (
                    _make_mc_critic_q1_carry(batch)
                    if use_mc
                    else _make_critic_q1_carry(batch)
                )
            if carry2 is None:
                carry2 = (
                    _make_mc_critic_q2_carry(batch)
                    if use_mc
                    else _make_critic_q2_carry(batch)
                )
            new_c1, mem_out1 = mem1(fe1(x), carry1)
            new_c2, mem_out2 = mem2(fe2(x), carry2)
            q1 = head1(jnp.concat((mem_out1, actions), axis=-1))  # (batch, 1)
            q2 = head2(jnp.concat((mem_out2, actions), axis=-1))  # (batch, 1)
            return jnp.concat([q1, q2], axis=-1), new_c1, new_c2  # (batch, 2)

        def run_critic_scan(
            critic: TwinCriticState,
            xs: jax.Array,
            actions: jax.Array,
            use_mc: bool = False,
            initial_carry_q1=None,
            initial_carry_q2=None,
            dones: jax.Array | None = None,
            return_input_carries: bool = False,
        ) -> tuple[any, any, jax.Array]:
            """Run both continuous critic branches over an observation sequence.

            Args:
                critic: Twin-critic network state.
                xs: Observation sequence of shape ``(batch, T, *obs_shape)``.
                actions: Action sequence of shape ``(batch, T, action_dim)``.
                use_mc: If True, use MC critic graph definitions.
                initial_carry_q1: Initial carry for Q1.  ``None`` → zero carry.
                initial_carry_q2: Initial carry for Q2.  ``None`` → zero carry.
                dones: Optional terminal mask ``(batch, T)``; the carry leaving
                    a terminal step is zeroed before the next step.
                return_input_carries: If True, also return per-step input
                    carries for both branches.

            Returns:
                ``(final_carry_q1, final_carry_q2, qs)`` where ``qs`` has
                shape ``(batch, T, 2)``.  When ``return_input_carries=True``,
                two additional elements ``(input_carries_q1, input_carries_q2)``
                are appended.
            """
            fe1 = nnx.merge(
                mc_critic_q1_fe_graph if use_mc else critic_q1_fe_graph, critic.q1.fe
            )
            mem1 = nnx.merge(
                mc_critic_q1_memory_graph if use_mc else critic_q1_memory_graph,
                critic.q1.memory,
                mc_critic_q1_memory_rngs if use_mc else critic_q1_memory_rngs,
            )
            head1 = nnx.merge(
                mc_critic_q1_head_graph if use_mc else critic_q1_head_graph,
                critic.q1.head,
            )
            fe2 = nnx.merge(
                mc_critic_q2_fe_graph if use_mc else critic_q2_fe_graph, critic.q2.fe
            )
            mem2 = nnx.merge(
                mc_critic_q2_memory_graph if use_mc else critic_q2_memory_graph,
                critic.q2.memory,
                mc_critic_q2_memory_rngs if use_mc else critic_q2_memory_rngs,
            )
            head2 = nnx.merge(
                mc_critic_q2_head_graph if use_mc else critic_q2_head_graph,
                critic.q2.head,
            )
            batch, T = xs.shape[0], xs.shape[1]
            flat_xs = xs.reshape(batch * T, *xs.shape[2:])
            feats1 = fe1(flat_xs).reshape(batch, T, -1)
            feats2 = fe2(flat_xs).reshape(batch, T, -1)
            if initial_carry_q1 is None:
                initial_carry_q1 = (
                    _make_mc_critic_q1_carry(batch)
                    if use_mc
                    else _make_critic_q1_carry(batch)
                )
            if initial_carry_q2 is None:
                initial_carry_q2 = (
                    _make_mc_critic_q2_carry(batch)
                    if use_mc
                    else _make_critic_q2_carry(batch)
                )
            if return_input_carries:
                fc1, mem_outs1, in_c1 = mem1.scan(
                    feats1, initial_carry_q1, dones, return_input_carries=True
                )
                fc2, mem_outs2, in_c2 = mem2.scan(
                    feats2, initial_carry_q2, dones, return_input_carries=True
                )
            else:
                fc1, mem_outs1 = mem1.scan(feats1, initial_carry_q1, dones)
                fc2, mem_outs2 = mem2.scan(feats2, initial_carry_q2, dones)
            flat_actions = actions.reshape(batch * T, -1)
            flat_mem1 = mem_outs1.reshape(batch * T, -1)
            flat_mem2 = mem_outs2.reshape(batch * T, -1)
            q1 = head1(jnp.concat((flat_mem1, flat_actions), axis=-1)).reshape(
                batch, T, 1
            )
            q2 = head2(jnp.concat((flat_mem2, flat_actions), axis=-1)).reshape(
                batch, T, 1
            )
            qs = jnp.concat([q1, q2], axis=-1)  # (batch, T, 2)
            if return_input_carries:
                return fc1, fc2, qs, in_c1, in_c2
            return fc1, fc2, qs

        def get_q_both(
            critic: TwinCriticState,
            x: jax.Array,
            carry1: any,
            carry2: any,
            actions: jax.Array,
            use_mc: bool = False,
        ) -> Tuple[jax.Array, jax.Array, any, any]:
            """Per-branch Q-values for continuous actions.

            Returns:
                ``(q1, q2, new_carry1, new_carry2)`` each Q of shape ``(batch,)``.
            """
            q, nc1, nc2 = run_critic(critic, x, actions, carry1, carry2, use_mc)
            return q[:, 0], q[:, 1], nc1, nc2

        def get_q(
            critic: TwinCriticState,
            x: jax.Array,
            carry1: any,
            carry2: any,
            actions: jax.Array,
            use_mc: bool = False,
        ) -> tuple[jax.Array, any, any]:
            """Return the conservative (min over twin) Q-value for each transition."""
            q1, q2, nc1, nc2 = get_q_both(critic, x, carry1, carry2, actions, use_mc)
            return jnp.minimum(q1, q2), nc1, nc2

        def get_dist_params(
            actor: NetworkState, x: jax.Array, actor_carry: any = None
        ) -> Tuple[jax.Array, jax.Array, any]:
            """Extract mean and std of the squashed Gaussian policy.

            The raw log-std output is tanh-squashed and rescaled into
            ``[LOG_STD_MIN, LOG_STD_MAX]`` for numerical stability.

            Returns:
                ``(mean, std, new_actor_carry)`` mean/std of shape ``(batch, action_dim)``.
            """
            if actor_carry is None:
                actor_carry = _make_actor_carry(x.shape[0])
            new_actor_carry, dist_params = run_actor(actor, x, actor_carry)
            mean, log_std = (
                dist_params[..., :action_dim],
                dist_params[..., action_dim:],
            )
            log_std = jnp.tanh(log_std)
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
            std = jnp.exp(log_std)
            return mean, std, new_actor_carry

        def sample_action_logprob(
            actor: NetworkState,
            x: jax.Array,
            key: jax.Array,
            actor_carry: any = None,
        ) -> Tuple[jax.Array, jax.Array, jax.Array, any]:
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
            mean, std, new_actor_carry = get_dist_params(actor, x, actor_carry)
            unsquashed_action = jax.random.normal(key, mean.shape) * std + mean
            y_t = jnp.tanh(unsquashed_action)
            action = y_t * action_scale + action_bias
            log_prob = (
                -((unsquashed_action - mean) ** 2) / (2 * std**2)
                - 0.5 * jnp.log(2 * jnp.pi)
                - jnp.log(std)
                - jnp.log(action_scale * (1 - y_t**2) + 1e-6)
            )
            return action, unsquashed_action, log_prob.sum(axis=-1), new_actor_carry

        @jax.jit
        def get_importance_ratios(
            actor: NetworkState,
            obs: jax.Array,
            actions: jax.Array,
            behaviour_probs: jax.Array,
            actor_carry: any = None,
        ) -> tuple[jax.Array, any]:
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
            mean, std, new_actor_carry = get_dist_params(
                actor, obs, actor_carry
            )  # (batch, action_dim)
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
            return jnp.exp(log_prob) / behaviour_probs, new_actor_carry

        def get_v(
            actor: NetworkState,
            critic: TwinCriticState,
            alpha: jax.Array,
            x: jax.Array,
            actor_carry: any,
            q_carry1: any,
            q_carry2: any,
            key: jax.Array,
            include_entropy: bool = True,
            include_log: bool = False,
            use_mc: bool = False,
        ) -> Tuple:
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
            action, _unsquashed, logprob, new_a_c = sample_action_logprob(
                actor, x, key, actor_carry
            )
            q, new_q1, new_q2 = get_q(
                critic, x, q_carry1, q_carry2, action, use_mc
            )
            new_carries = (new_a_c, new_q1, new_q2)
            if include_entropy:
                v = q - alpha * logprob
                if include_log:
                    return (
                        v,
                        new_carries,
                        {
                            "q": q.mean(),
                            "entropy": -logprob.mean(),
                        },
                    )
                return v, new_carries
            else:
                return q, new_carries

        def get_entropy(actor, obs, actor_carry, key):
            _action, _unsquashed, logprob, _new_a_c = sample_action_logprob(
                actor, obs, key, actor_carry
            )
            return -logprob

        @partial(jax.jit, static_argnames=["deterministic", "return_unsquashed"])
        def predict(
            iqlearn: IQLearnState,
            obs: jax.Array,
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
            obs_batch = jnp.expand_dims(obs, 0)
            mean, std, _new_carry = get_dist_params(iqlearn.actor, obs_batch)
            if deterministic:
                unsquashed_action = mean
            else:
                unsquashed_action = jax.random.normal(key, mean.shape) * std + mean
            y_t = jnp.tanh(unsquashed_action)
            action = y_t * action_scale + action_bias
            if return_unsquashed:
                return action[0], unsquashed_action[0]
            return action[0]

    # ------------------------------------------------------------------
    # Loss functions and other helpers which are structurally identical
    # for both action space types; differences are absorbed by the
    # action-space-specific helpers above.
    # ------------------------------------------------------------------

    def calculate_td_lambda(
        actor: NetworkState,
        critic: TwinCriticState,
        alpha: jax.Array,
        key: jax.Array,
        obs_seq: jax.Array,
        action_seq: jax.Array,
        behaviour_seq: jax.Array,
        reward_seq: jax.Array,
        done_seq: jax.Array,
        actor_carry,
        cq1_carry,
        cq2_carry,
    ):
        """Batched V-trace TD(λ) return with R2D2-style carry threading.

        Inputs are time-batched: all sequence tensors have leading shape
        ``(batch, N + 1)`` where ``N = params.lambda_truncation``.  The
        function runs a :func:`jax.lax.scan` across the ``N + 1`` timesteps
        so the actor and critic carries are advanced through recurrent memory
        rather than started fresh at every step.  Terminal steps reset the
        outgoing carry to zero (R2D2 §2.1), and the cumulative IS weight is
        masked past the first terminal.

        Args:
            actor: Current actor network state.
            critic: MC critic target state (``state.mc_critic_target``).
                Evaluated with ``use_mc=True``.
            alpha: Entropy temperature.
            key: JAX PRNG key for entropy / value samples inside the scan.
            obs_seq: ``(batch, N + 1, *obs_shape)``.
            action_seq: ``(batch, N + 1, *action_shape)``.  For discrete
                spaces, float32 action indices.  For continuous, the
                unsquashed Gaussian samples stored under
                ``unsquashed_action_key``.
            behaviour_seq: ``(batch, N + 1)``.
            reward_seq: ``(batch, N + 1)``.
            done_seq: ``(batch, N + 1)`` float32 terminal mask.
            actor_carry: ``(batch, *carry_shape)`` actor carry at the start
                of the window (post-burn-in).
            cq1_carry, cq2_carry: matching critic carries for the MC twin.

        Returns:
            ``(td, p_k, new_actor_carry, new_cq1, new_cq2)`` where ``td``
            has shape ``(batch,)`` and ``p_k`` is the clipped cumulative
            importance weight (shape ``(batch,)``).  The returned carries
            are the states at the *end* of the scanned window — useful
            when chaining consecutive windows.
        """
        N1 = params.lambda_truncation + 1
        batch = obs_seq.shape[0]

        obs_T = jnp.swapaxes(obs_seq, 0, 1)
        act_T = jnp.swapaxes(action_seq, 0, 1)
        beh_T = jnp.swapaxes(behaviour_seq, 0, 1)
        done_T = jnp.swapaxes(done_seq, 0, 1)

        def reset_on_done(c, done_t):
            d = done_t.reshape((-1,) + (1,) * (c.ndim - 1))
            return jnp.where(d, jnp.zeros_like(c), c)

        def step(carries, xs):
            a_c, q1c, q2c, key_c = carries
            obs_t, act_t, beh_t, done_t = xs
            key_c, k_ent, k_v = jax.random.split(key_c, 3)

            if is_discrete:
                c_t, new_a_c = get_importance_ratios(
                    actor, obs_t, a_c, act_t, beh_t
                )
                ent_t = get_entropy(actor, obs_t, a_c, k_ent)
                v_t, (_nac_unused, new_q1c, new_q2c) = get_v(
                    actor,
                    critic,
                    alpha,
                    obs_t,
                    a_c,
                    q1c,
                    q2c,
                    k_v,
                    include_entropy=True,
                    use_mc=True,
                )
            else:
                c_t, new_a_c = get_importance_ratios(
                    actor, obs_t, act_t, beh_t, a_c
                )
                ent_t = get_entropy(actor, obs_t, a_c, k_ent)
                v_t, (_nac_unused, new_q1c, new_q2c) = get_v(
                    actor,
                    critic,
                    alpha,
                    obs_t,
                    a_c,
                    q1c,
                    q2c,
                    k_v,
                    include_entropy=True,
                    use_mc=True,
                )

            new_a_c = reset_on_done(new_a_c, done_t)
            new_q1c = reset_on_done(new_q1c, done_t)
            new_q2c = reset_on_done(new_q2c, done_t)

            return (new_a_c, new_q1c, new_q2c, key_c), (c_t, ent_t, v_t)

        (final_a, final_q1, final_q2, _), (c_k_T, ent_T, v_T) = jax.lax.scan(
            step,
            (actor_carry, cq1_carry, cq2_carry, key),
            (obs_T, act_T, beh_T, done_T),
        )

        c_k = jnp.swapaxes(c_k_T, 0, 1)  # (batch, N+1)
        ent = jnp.swapaxes(ent_T, 0, 1)  # (batch, N+1)
        v = jnp.swapaxes(v_T, 0, 1)      # (batch, N+1)

        p_k = jnp.clip(jnp.prod(c_k, axis=1), max=1.0)  # (batch,)

        gammas = params.gamma ** jnp.arange(N1)                       # (N+1,)
        lambdas = params.lam ** jnp.arange(params.lambda_truncation)  # (N,)

        # done_mask[k] = 1 if no terminal in steps 0..k-1, else 0.
        cum_term = jnp.maximum.accumulate(
            done_seq[:, :-1].astype(jnp.float32), axis=1
        )  # (batch, N)
        leading = jnp.zeros((batch, 1), dtype=jnp.float32)
        done_mask = 1.0 - jnp.concatenate([leading, cum_term], axis=1)  # (batch, N+1)

        rewards_adj = reward_seq[:, :-1] + alpha * ent[:, :-1]  # (batch, N)
        rn = jnp.cumsum(
            gammas[:-1][None, :] * rewards_adj * done_mask[:, :-1], axis=1
        )  # (batch, N)
        rn = rn + gammas[1:][None, :] * done_mask[:, 1:] * v[:, 1:]

        td = (1.0 - params.lam) * jnp.sum(lambdas[None, :] * rn, axis=1)
        return td, p_k, final_a, final_q1, final_q2

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
        sample: SequenceSample,
        alpha: jax.Array,
        key: jax.Array,
        mc_critic_target: TwinCriticState | None = None,
    ) -> Tuple[jax.Array, dict]:
        """Actor loss: maximise the soft state value V(s) = Q(s,a) - α log π(a|s).

        Returns:
            ``(scalar_loss, metrics)`` where metrics contains ``"q"``,
            ``"entropy"``, ``"v"``, and (when discrepancy enabled)
            ``"lambda_discrepancy_loss"``.
        """
        use_disc = (
            mc_critic_target is not None and params.lambda_discrepancy_coef > 0.0
        )
        key_v = key
        obs, act, mask = sample.obs, sample.act, sample.mask
        burn_ac, burn_cq1, burn_cq2 = sample.burn_ac, sample.burn_cq1, sample.burn_cq2
        seq_idx = sample.seq_idx
        xs = (
            jnp.swapaxes(obs, 0, 1),
            jnp.swapaxes(act, 0, 1),
            jnp.swapaxes(mask, 0, 1),
        )

        if use_disc:
            critic_sg = jax.tree.map(jax.lax.stop_gradient, critic)
            mc_target_sg = jax.tree.map(jax.lax.stop_gradient, mc_critic_target)

        def scan_fun(carry, x):
            (actor_carry, q_carry1, q_carry2,
             mc_q1c, mc_q2c) = carry
            input_actor_carry = actor_carry  # captured before stop_gradient
            actor_carry = jax.lax.stop_gradient(actor_carry)
            q_carry1 = jax.lax.stop_gradient(q_carry1)
            q_carry2 = jax.lax.stop_gradient(q_carry2)
            mc_q1c = jax.lax.stop_gradient(mc_q1c)
            mc_q2c = jax.lax.stop_gradient(mc_q2c)
            obs, act_t, mask = x
            v, new_carries, metrics = get_v(
                actor,
                critic,
                alpha,
                obs,
                actor_carry,
                q_carry1,
                q_carry2,
                key_v,
                include_entropy=True,
                include_log=True,
            )
            metrics = {**metrics, "v": v.mean()}
            step_loss = -(v * jnp.reshape(mask, -1)).mean()
            new_a_c, new_q1c, new_q2c = new_carries
            if use_disc:
                q_sac, _, _ = get_q(critic_sg, obs, q_carry1, q_carry2, act_t)
                q_mc, new_mc_q1c, new_mc_q2c = get_q(
                    mc_target_sg, obs, mc_q1c, mc_q2c, act_t, use_mc=True
                )
                H = get_entropy(actor, obs, actor_carry, key_v)
                delta = (
                    jax.lax.stop_gradient(q_sac)
                    + alpha * H
                    - jax.lax.stop_gradient(q_mc)
                )
                disc = optax.losses.huber_loss(
                    delta, delta=params.lambda_discrepancy_delta
                )
                disc_loss = (disc * jnp.reshape(mask, -1)).mean()
                step_loss = step_loss + params.lambda_discrepancy_coef * disc_loss
                metrics = {**metrics, "lambda_discrepancy_loss": disc_loss}
            else:
                new_mc_q1c, new_mc_q2c = mc_q1c, mc_q2c
            return (new_a_c, new_q1c, new_q2c, new_mc_q1c, new_mc_q2c), (
                step_loss,
                metrics,
                jax.lax.stop_gradient(input_actor_carry),
            )

        _bs = obs.shape[0]
        if use_disc:
            init_mc1 = sample.burn_mc_cq1_tgt
            init_mc2 = sample.burn_mc_cq2_tgt
        else:
            # Placeholder — never read inside scan when use_disc=False.
            init_mc1 = burn_cq1
            init_mc2 = burn_cq2
        _, (losses, metrics, actor_input_carries_T) = jax.lax.scan(
            scan_fun,
            (burn_ac, burn_cq1, burn_cq2, init_mc1, init_mc2),
            xs,
        )

        metrics.update({"loss_actor": losses})
        # Carry-refresh payload: input actor carry per scan step → slot
        # seq_idx[:, B + t].  Empty when refresh disabled to avoid extra work.
        if params.refresh_stored_carries:
            B = params.burn_in_length
            SL = params.sequence_length + 1
            actor_input_carries = jax.tree.map(
                lambda c: jnp.swapaxes(c, 0, 1), actor_input_carries_T
            )  # leaves: (batch, SL, *carry)
            refresh = {
                actor_carry_key: (seq_idx[:, B:B + SL], actor_input_carries),
            }
        else:
            refresh = {}
        return losses.mean(), (
            jax.tree.map(lambda x: x.mean(), metrics),
            refresh,
        )

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

        _bs = sample.this_info[obs_key].shape[0]
        _z_a = _make_actor_carry(_bs)
        _z_q1 = _make_critic_q1_carry(_bs)
        _z_q2 = _make_critic_q2_carry(_bs)
        q_values, _, _ = get_q(
            critic,
            sample.this_info[obs_key],
            _z_q1,
            _z_q2,
            sample.this_info[action_key],
        )
        v_values, _ = get_v(
            actor_target,
            critic,
            alpha,
            sample.this_info[obs_key],
            _z_a,
            _z_q1,
            _z_q2,
            key_v,
            include_entropy=True,
        )
        next_v_values, _ = get_v(
            actor_target,
            critic_target,
            alpha,
            sample.next_info[obs_key],
            _z_a,
            _z_q1,
            _z_q2,
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

    def loss_critic_sac(
        actor_target: NetworkState,
        critic: TwinCriticState,
        critic_target: TwinCriticState,
        sample: SequenceSample,
        alpha: jax.Array,
        key: jax.Array,
        mc_critic_target: TwinCriticState | None = None,
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
            sample: Pre-sampled sequence with burned-in carries.
            alpha: Current entropy temperature.
            key: JAX PRNG key for action sampling in continuous V.

        Returns:
            ``(scalar_loss, metrics)`` where metrics contains
            ``"critic_loss"`` and ``"target_q"``.
        """
        key_v = key
        obs, act, rew, done, mask, seq_idx = (
            sample.obs, sample.act, sample.rew, sample.done, sample.mask, sample.seq_idx
        )
        burn_ac = sample.burn_ac_tgt
        burn_cq1, burn_cq2 = sample.burn_cq1, sample.burn_cq2
        burn_cq1_tgt, burn_cq2_tgt = sample.burn_cq1_tgt, sample.burn_cq2_tgt
        burn_mc_cq1, burn_mc_cq2 = sample.burn_mc_cq1_tgt, sample.burn_mc_cq2_tgt
        # The buffer sampler returns SL = sequence_length + 1 timesteps so
        # that V(s_{t+1}) is available as the bootstrap target for transition
        # t.  The previous implementation evaluated V on s_t itself, which
        # produced target_q = r_t + γ V(s_t) and broke value propagation —
        # Q simply regressed onto its own current state.

        # Pass 1: target V over the whole sequence.  Carries thread through
        # actor_target and critic_target so V(s_{t+1}) sees the recurrent
        # state produced by s_0..s_t.  Refactored to one batched
        # ``run_actor_scan`` + one ``run_critic_scan`` per network instead of
        # ``T`` sequential per-step calls — same numerics, far fewer kernel
        # launches.
        SL_static = params.sequence_length + 1
        _bs = obs.shape[0]
        if is_discrete:
            # Discrete V is a closed-form expectation; no PRNG sampling.
            _, logits_T = run_actor_scan(
                actor_target, obs, jax.lax.stop_gradient(burn_ac)
            )  # (B, SL, A)
            _, _, q_twin_target_T = run_critic_scan(
                critic_target,
                obs,
                False,
                jax.lax.stop_gradient(burn_cq1_tgt),
                jax.lax.stop_gradient(burn_cq2_tgt),
            )  # (B, SL, A, 2)
            probs_T = jax.nn.softmax(logits_T, axis=-1)
            log_probs_T = jax.nn.log_softmax(logits_T, axis=-1)
            q_min_target_T = jnp.minimum(
                q_twin_target_T[..., 0], q_twin_target_T[..., 1]
            )
            entropy_T = -(probs_T * log_probs_T).sum(-1)
            v = (probs_T * q_min_target_T).sum(-1) + alpha * entropy_T  # (B, SL)
        else:
            # Continuous V uses a sampled action per step.  Batch the actor
            # forward (FE+memory+head); then a thin ``jax.lax.scan`` mirrors
            # the original ``v_scan`` exactly — same key chain, same critic
            # carry threading via ``run_critic`` per-step.  This is byte-
            # identical to the original because the critic uses untouched
            # ``run_critic`` (no batched FE), only the actor forward changes,
            # and ``run_actor_scan`` produces bit-identical mean/std (verified
            # by the discrete equivalence path).
            _, dist_params_T = run_actor_scan(
                actor_target, obs, jax.lax.stop_gradient(burn_ac)
            )  # (B, SL, 2*action_dim)
            mean_T = dist_params_T[..., :action_dim]
            log_std_T = dist_params_T[..., action_dim:]
            log_std_T = jnp.tanh(log_std_T)
            log_std_T = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std_T + 1)
            std_T = jnp.exp(log_std_T)

            obs_T_v = jnp.swapaxes(obs, 0, 1)
            mean_T_t = mean_T.swapaxes(0, 1)  # (SL, B, ad)
            std_T_t = std_T.swapaxes(0, 1)

            def _v_scan_cont(carry, x):
                q1c, q2c, k = carry
                obs_t, m_t, s_t = x
                k, k_step = jax.random.split(k)
                u = jax.random.normal(k_step, m_t.shape) * s_t + m_t
                y = jnp.tanh(u)
                a = y * action_scale + action_bias
                log_prob = (
                    -((u - m_t) ** 2) / (2 * s_t**2)
                    - 0.5 * jnp.log(2 * jnp.pi)
                    - jnp.log(s_t)
                    - jnp.log(action_scale * (1 - y**2) + 1e-6)
                ).sum(-1)
                q_twin_t, new_q1c, new_q2c = run_critic(
                    critic_target, obs_t, a, q1c, q2c, False
                )
                q_t = jnp.minimum(q_twin_t[:, 0], q_twin_t[:, 1])
                v_t = q_t - alpha * log_prob
                return (new_q1c, new_q2c, k), v_t

            _, v_T = jax.lax.scan(
                _v_scan_cont,
                (
                    jax.lax.stop_gradient(burn_cq1_tgt),
                    jax.lax.stop_gradient(burn_cq2_tgt),
                    key_v,
                ),
                (obs_T_v, mean_T_t, std_T_t),
            )
            v = jnp.swapaxes(v_T, 0, 1)  # (B, SL)

        # Pass 2: online Q at each timestep with online critic carries.
        # Refactored to one batched ``run_critic_scan`` (with optional
        # per-step input-carry capture for the buffer refresh feature).
        _refresh = params.refresh_stored_carries
        if is_discrete:
            if _refresh:
                _, _, q_twin_T, cq1_input_T, cq2_input_T = run_critic_scan(
                    critic, obs, False, burn_cq1, burn_cq2,
                    return_input_carries=True,
                )
            else:
                _, _, q_twin_T = run_critic_scan(
                    critic, obs, False, burn_cq1, burn_cq2,
                )
                cq1_input_T = cq2_input_T = None
            ai = jnp.round(act.reshape(_bs, SL_static)).astype(jnp.int32)  # (B, SL)
            bi = jnp.arange(_bs)[:, None]
            ti = jnp.arange(SL_static)[None, :]
            q1 = q_twin_T[bi, ti, ai, 0]  # (B, SL)
            q2 = q_twin_T[bi, ti, ai, 1]
        else:
            if _refresh:
                _, _, q_twin_T, cq1_input_T, cq2_input_T = run_critic_scan(
                    critic, obs, act, False, burn_cq1, burn_cq2,
                    return_input_carries=True,
                )
            else:
                _, _, q_twin_T = run_critic_scan(
                    critic, obs, act, False, burn_cq1, burn_cq2,
                )
                cq1_input_T = cq2_input_T = None
            q1 = q_twin_T[..., 0]
            q2 = q_twin_T[..., 1]

        if _refresh:
            cq1_input_T = jax.lax.stop_gradient(cq1_input_T)
            cq2_input_T = jax.lax.stop_gradient(cq2_input_T)

        # Bellman target uses the NEXT-step V; loss is over the first SL-1
        # transitions (the last slot only contributes its V to the target).
        target_q = jax.lax.stop_gradient(
            rew[:, :-1] + params.gamma * (1.0 - done[:, :-1]) * v[:, 1:]
        )
        m = mask[:, :-1]
        denom = jnp.maximum(m.sum(), 1.0)
        loss = 0.5 * (
            ((q1[:, :-1] - target_q) ** 2 * m).sum() / denom
            + ((q2[:, :-1] - target_q) ** 2 * m).sum() / denom
        )
        metrics = {
            "critic_loss": loss,
            "target_q": target_q.mean(),
            "target_q_std": target_q.std(),
            "target_q_max": target_q.max(),
            "target_q_min": target_q.min(),
            "sac_q1_mean": q1[:, :-1].mean(),
            "sac_q2_mean": q2[:, :-1].mean(),
            "sac_q_twin_gap_mean": jnp.abs(q1[:, :-1] - q2[:, :-1]).mean(),
        }

        # Lambda-discrepancy auxiliary loss: pulls Q_sac + α·H toward Q_mc.
        # Only the online SAC critic moves; everything else is stop_gradient.
        # Refactored: q_mc and entropy both produced by single batched scans
        # over the sequence rather than per-step lax.scan calls.
        if mc_critic_target is not None and params.lambda_discrepancy_coef > 0.0:
            mc_target_sg = jax.tree.map(jax.lax.stop_gradient, mc_critic_target)
            actor_target_sg = jax.tree.map(jax.lax.stop_gradient, actor_target)

            # Q_mc(s, a_taken) over the whole sequence, primed with burned-in carries.
            if is_discrete:
                _, _, q_mc_twin_T = run_critic_scan(
                    mc_target_sg, obs, True,
                    jax.lax.stop_gradient(burn_mc_cq1),
                    jax.lax.stop_gradient(burn_mc_cq2),
                )  # (B, SL, A, 2)
                ai = jnp.round(act.reshape(_bs, SL_static)).astype(jnp.int32)
                bi = jnp.arange(_bs)[:, None]
                ti = jnp.arange(SL_static)[None, :]
                q_mc = jnp.minimum(
                    q_mc_twin_T[bi, ti, ai, 0],
                    q_mc_twin_T[bi, ti, ai, 1],
                )  # (B, SL)
            else:
                _, _, q_mc_twin_T = run_critic_scan(
                    mc_target_sg, obs, act, True,
                    jax.lax.stop_gradient(burn_mc_cq1),
                    jax.lax.stop_gradient(burn_mc_cq2),
                )  # (B, SL, 2)
                q_mc = jnp.minimum(
                    q_mc_twin_T[..., 0], q_mc_twin_T[..., 1]
                )  # (B, SL)

            # Entropy H(π_target) over the sequence.  Replicates the
            # original ``h_scan``: discrete is a closed-form expectation;
            # continuous uses a single sample with the SAME ``key_v`` at
            # every timestep (matches per-step ``get_entropy(..., key_v)``
            # in the original code, which never re-split the key).
            _, dist_T_h = run_actor_scan(actor_target_sg, obs, burn_ac)
            if is_discrete:
                probs_at_T = jax.nn.softmax(dist_T_h, axis=-1)
                log_probs_at_T = jax.nn.log_softmax(dist_T_h, axis=-1)
                H = -(probs_at_T * log_probs_at_T).sum(-1)  # (B, SL)
            else:
                mean_at_T = dist_T_h[..., :action_dim]
                log_std_at_T = jnp.tanh(dist_T_h[..., action_dim:])
                log_std_at_T = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
                    log_std_at_T + 1
                )
                std_at_T = jnp.exp(log_std_at_T)
                # Same (B, action_dim) noise at every t — matches buggy
                # original behaviour.  Re-introducing per-step keys here
                # would change numerics.
                noise = jax.random.normal(key_v, (_bs, action_dim))
                noise_T = jnp.broadcast_to(noise[:, None, :], mean_at_T.shape)
                u_T = noise_T * std_at_T + mean_at_T
                y_T = jnp.tanh(u_T)
                log_prob_T = (
                    -((u_T - mean_at_T) ** 2) / (2 * std_at_T**2)
                    - 0.5 * jnp.log(2 * jnp.pi)
                    - jnp.log(std_at_T)
                    - jnp.log(action_scale * (1 - y_T**2) + 1e-6)
                ).sum(-1)
                H = -log_prob_T  # (B, SL)

            q_sac = jnp.minimum(q1, q2)  # (bs, SL) — gradient flows through critic
            delta = (
                q_sac
                + alpha * jax.lax.stop_gradient(H)
                - jax.lax.stop_gradient(q_mc)
            )
            disc = optax.losses.huber_loss(
                delta, delta=params.lambda_discrepancy_delta
            )
            disc_loss = (disc * mask).sum() / jnp.maximum(mask.sum(), 1.0)
            loss = loss + params.lambda_discrepancy_coef * disc_loss
            metrics["lambda_discrepancy_loss"] = disc_loss
            metrics["critic_loss"] = loss
            m_full = mask
            m_denom = jnp.maximum(m_full.sum(), 1.0)
            metrics["disc_delta_abs_mean"] = (jnp.abs(delta) * m_full).sum() / m_denom
            metrics["disc_delta_abs_max"] = jnp.abs(delta).max()
            metrics["disc_q_sac_mean"] = (q_sac * m_full).sum() / m_denom
            metrics["disc_q_mc_mean"] = (q_mc * m_full).sum() / m_denom
            metrics["disc_entropy_mean"] = (H * m_full).sum() / m_denom

        # Carry-refresh payload: input online-critic carries per scan step.
        # ``run_critic_scan`` with ``return_input_carries=True`` already returns
        # them batch-major, so no transpose is needed.
        if params.refresh_stored_carries:
            B = params.burn_in_length
            SL = params.sequence_length + 1
            refresh = {
                critic_q1_carry_key: (seq_idx[:, B:B + SL], cq1_input_T),
                critic_q2_carry_key: (seq_idx[:, B:B + SL], cq2_input_T),
            }
        else:
            refresh = {}
        return loss, (metrics, refresh)

    def loss_mc_critic_sac(
        actor: NetworkState,
        mc_critic: TwinCriticState,
        mc_critic_target: TwinCriticState,
        online_buf: Buffer,
        sample: SequenceSample,
        alpha: jax.Array,
        key: jax.Array,
        critic_target: TwinCriticState | None = None,
    ) -> Tuple[jax.Array, dict]:
        """MC-critic Bellman MSE loss with R2D2 burn-in and TD(λ) targets.

        Scans across the pre-sampled sequence.  At each step a TD(λ) look-ahead
        window is pulled directly from ``online_buf`` (still needed for
        random-access indexing).  The MC-critic is trained to match the
        V-trace TD(λ) target under the stored IS clipping coefficient.
        """
        key_scan = key

        obs, act, rew, done, mask, seq_idx = (
            sample.obs, sample.act, sample.rew, sample.done, sample.mask, sample.seq_idx
        )
        burn_ac = sample.burn_ac
        burn_cq1, burn_cq2 = sample.burn_mc_cq1, sample.burn_mc_cq2
        burn_cq1_tgt, burn_cq2_tgt = sample.burn_mc_cq1_tgt, sample.burn_mc_cq2_tgt

        B = params.burn_in_length
        N1 = params.lambda_truncation + 1
        buf_size = online_buf.info[obs_key].shape[0]

        obs_T = jnp.swapaxes(obs, 0, 1)     # (SL, batch, *obs)
        act_T = jnp.swapaxes(act, 0, 1)     # (SL, batch, *)
        mask_T = jnp.swapaxes(mask, 0, 1)   # (SL, batch)
        done_T = jnp.swapaxes(done, 0, 1)   # (SL, batch)
        # Per-sequence-step start index for the TD(λ) look-ahead window.
        seq_start_T = jnp.swapaxes(seq_idx[:, B:], 0, 1)  # (SL, batch)

        def reset_on_done(c, done_t):
            d = done_t.reshape((-1,) + (1,) * (c.ndim - 1))
            return jnp.where(d, jnp.zeros_like(c), c)

        use_disc = (
            critic_target is not None and params.lambda_discrepancy_coef > 0.0
        )
        if use_disc:
            critic_target_sg = jax.tree.map(jax.lax.stop_gradient, critic_target)
            actor_sg = jax.tree.map(jax.lax.stop_gradient, actor)

        def scan_fun(carries, xs):
            actor_carry, q1c, q2c, tgt_q1c, tgt_q2c, sac_q1c, sac_q2c, key_c = carries
            input_q1c, input_q2c = q1c, q2c  # captured before stop_gradient
            obs_t, act_t, mask_t, done_t, start_t = xs
            key_c, k_td, k_step = jax.random.split(key_c, 3)

            actor_carry_sg = jax.lax.stop_gradient(actor_carry)
            q1c_sg = jax.lax.stop_gradient(q1c)
            q2c_sg = jax.lax.stop_gradient(q2c)
            tgt_q1c_sg = jax.lax.stop_gradient(tgt_q1c)
            tgt_q2c_sg = jax.lax.stop_gradient(tgt_q2c)

            # Build the look-ahead window (circular) starting at start_t.
            td_idx = (start_t[:, None] + jnp.arange(N1)[None, :]) % buf_size
            td_obs = online_buf.info[obs_key][td_idx]
            td_act_key = (
                unsquashed_action_key if not is_discrete else action_key
            )
            td_act = online_buf.info[td_act_key][td_idx]
            td_beh = online_buf.info[behaviour_key][td_idx]
            td_rew = online_buf.info[reward_key][td_idx]
            td_done = online_buf.info[terminated_key][td_idx]
            # Virtual terminal at write head — slot[pos % size] holds stale
            # previous-cycle data; truncate the lookahead at this boundary.
            write_head = online_buf.pos % buf_size
            td_done = jnp.where(td_idx == write_head, 1.0, td_done)

            td, p_k, _fa, _fq1, _fq2 = calculate_td_lambda(
                actor,
                mc_critic_target,
                alpha,
                k_td,
                td_obs,
                td_act,
                td_beh,
                td_rew,
                td_done,
                actor_carry_sg,
                tgt_q1c_sg,
                tgt_q2c_sg,
            )
            target_q = jax.lax.stop_gradient(td)
            coefs = jax.lax.stop_gradient(p_k)

            # Online MC critic Q for gradient. Carries advance by one step.
            if is_discrete:
                q1, q2, new_q1c, new_q2c = get_q_both(
                    mc_critic, obs_t, q1c, q2c, act_t, use_mc=True
                )
            else:
                q1, q2, new_q1c, new_q2c = get_q_both(
                    mc_critic, obs_t, q1c, q2c, act_t, use_mc=True
                )

            # Advance actor carry by 1 step on obs_t (for the next outer step).
            new_actor_carry, _ = run_actor(actor, obs_t, actor_carry)

            # Advance mc_critic_target carry by 1 step — keeps it in sync with
            # the sequence so calculate_td_lambda starts from the right state.
            _, _, new_tgt_q1c, new_tgt_q2c = get_q_both(
                mc_critic_target, obs_t, tgt_q1c, tgt_q2c, act_t, use_mc=True
            )

            new_actor_carry = reset_on_done(new_actor_carry, done_t)
            new_q1c = reset_on_done(new_q1c, done_t)
            new_q2c = reset_on_done(new_q2c, done_t)
            new_tgt_q1c = reset_on_done(new_tgt_q1c, done_t)
            new_tgt_q2c = reset_on_done(new_tgt_q2c, done_t)

            loss_step = 0.5 * (
                jnp.mean(mask_t * coefs * (q1 - target_q) ** 2)
                + jnp.mean(mask_t * coefs * (q2 - target_q) ** 2)
            )

            disc_step = jnp.float32(0.0)
            if use_disc:
                # Q_sac at (s,a) under SAC critic target — stop gradient.
                # H_π(s) under stop-gradient actor.
                q_sac1, q_sac2, new_sac_q1c, new_sac_q2c = get_q_both(
                    critic_target_sg,
                    obs_t,
                    sac_q1c,
                    sac_q2c,
                    act_t,
                    use_mc=False,
                )
                q_sac_min = jnp.minimum(q_sac1, q_sac2)
                H_t = get_entropy(actor_sg, obs_t, actor_carry_sg, k_step)
                q_mc_min = jnp.minimum(q1, q2)  # gradient flows
                delta = (
                    jax.lax.stop_gradient(q_sac_min + alpha * H_t) - q_mc_min
                )
                disc = optax.losses.huber_loss(
                    delta, delta=params.lambda_discrepancy_delta
                )
                disc_step = (disc * coefs * mask_t).mean()
                loss_step = (
                    loss_step + params.lambda_discrepancy_coef * disc_step
                )
                new_sac_q1c = reset_on_done(new_sac_q1c, done_t)
                new_sac_q2c = reset_on_done(new_sac_q2c, done_t)
            else:
                new_sac_q1c, new_sac_q2c = sac_q1c, sac_q2c

            q_min_online = jnp.minimum(q1, q2)
            return (
                new_actor_carry,
                new_q1c,
                new_q2c,
                new_tgt_q1c,
                new_tgt_q2c,
                new_sac_q1c,
                new_sac_q2c,
                key_c,
            ), (
                loss_step,
                disc_step,
                jax.lax.stop_gradient(input_q1c),
                jax.lax.stop_gradient(input_q2c),
                jax.lax.stop_gradient(p_k),
                jax.lax.stop_gradient(target_q),
                jax.lax.stop_gradient(q_min_online),
            )

        _bs = obs.shape[0]
        if use_disc:
            init_sac_q1c = _make_critic_q1_carry(_bs)
            init_sac_q2c = _make_critic_q2_carry(_bs)
        else:
            init_sac_q1c = burn_cq1
            init_sac_q2c = burn_cq2
        _, (
            losses,
            disc_losses,
            mcq1_inputs_T,
            mcq2_inputs_T,
            p_k_T,
            target_q_T,
            q_min_T,
        ) = jax.lax.scan(
            scan_fun,
            (
                burn_ac, burn_cq1, burn_cq2,
                burn_cq1_tgt, burn_cq2_tgt,
                init_sac_q1c, init_sac_q2c,
                key_scan,
            ),
            (obs_T, act_T, mask_T, done_T, seq_start_T),
        )
        metrics = {"mc_critic_loss": losses.mean()}
        if use_disc:
            metrics["lambda_discrepancy_loss"] = disc_losses.mean()
        metrics["mc_pk_mean"] = p_k_T.mean()
        metrics["mc_pk_min"] = p_k_T.min()
        metrics["mc_pk_max"] = p_k_T.max()
        metrics["mc_pk_frac_collapsed"] = (p_k_T < 0.01).mean().astype(jnp.float32)
        metrics["mc_pk_frac_at_clip"] = (p_k_T >= 0.999).mean().astype(jnp.float32)
        metrics["mc_target_q_mean"] = target_q_T.mean()
        metrics["mc_target_q_std"] = target_q_T.std()
        metrics["mc_target_q_max"] = target_q_T.max()
        metrics["mc_target_q_min"] = target_q_T.min()
        metrics["mc_q_online_mean"] = q_min_T.mean()
        metrics["mc_td_residual_abs_mean"] = jnp.abs(target_q_T - q_min_T).mean()

        if params.refresh_stored_carries:
            SL = params.sequence_length + 1
            mcq1_inputs = jax.tree.map(
                lambda c: jnp.swapaxes(c, 0, 1), mcq1_inputs_T
            )
            mcq2_inputs = jax.tree.map(
                lambda c: jnp.swapaxes(c, 0, 1), mcq2_inputs_T
            )
            refresh = {
                mc_critic_q1_carry_key: (seq_idx[:, B:B + SL], mcq1_inputs),
                mc_critic_q2_carry_key: (seq_idx[:, B:B + SL], mcq2_inputs),
            }
        else:
            refresh = {}
        return losses.mean(), (metrics, refresh)

    # ------------------------------------------------------------------
    # Online helpers: environment interaction and SAC update
    # ------------------------------------------------------------------

    def run_env_step(sac: IQLearnState, env, env_params, env_state, key: jax.Array):
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
        obs_batch = jnp.expand_dims(obs, 0)
        # Snapshot pre-step carries for storage in the transition
        prev_actor_carry = sac.actor_online_carry
        prev_cq1_carry = sac.critic_q1_online_carry
        prev_cq2_carry = sac.critic_q2_online_carry
        if approximate_mc:
            prev_mcq1_carry = sac.mc_critic_q1_online_carry
            prev_mcq2_carry = sac.mc_critic_q2_online_carry
        # Run actor with current online carry to get action + updated carry
        new_actor_carry, actor_out = run_actor(
            sac.actor, obs_batch, sac.actor_online_carry
        )

        def _step_mem(fe_graph, fe_state, mem_graph, mem_state, mem_rngs, x, c):
            fe_mod = nnx.merge(fe_graph, fe_state)
            mem_mod = nnx.merge(mem_graph, mem_state, mem_rngs)
            new_c, _ = mem_mod(fe_mod(x), c)
            return new_c

        new_cq1_carry = _step_mem(
            critic_q1_fe_graph,
            sac.critic.q1.fe,
            critic_q1_memory_graph,
            sac.critic.q1.memory,
            critic_q1_memory_rngs,
            obs_batch,
            sac.critic_q1_online_carry,
        )
        new_cq2_carry = _step_mem(
            critic_q2_fe_graph,
            sac.critic.q2.fe,
            critic_q2_memory_graph,
            sac.critic.q2.memory,
            critic_q2_memory_rngs,
            obs_batch,
            sac.critic_q2_online_carry,
        )
        if approximate_mc:
            new_mcq1_carry = _step_mem(
                mc_critic_q1_fe_graph,
                sac.mc_critic.q1.fe,
                mc_critic_q1_memory_graph,
                sac.mc_critic.q1.memory,
                mc_critic_q1_memory_rngs,
                obs_batch,
                sac.mc_critic_q1_online_carry,
            )
            new_mcq2_carry = _step_mem(
                mc_critic_q2_fe_graph,
                sac.mc_critic.q2.fe,
                mc_critic_q2_memory_graph,
                sac.mc_critic.q2.memory,
                mc_critic_q2_memory_rngs,
                obs_batch,
                sac.mc_critic_q2_online_carry,
            )
        if is_discrete:
            logits = actor_out[0]
            action = jax.random.categorical(key_act, logits).astype(jnp.float32)
            prob = jax.nn.softmax(logits)[jnp.round(action).astype(jnp.int32)]
            env_action = jnp.round(action).astype(jnp.int32)
        else:
            dist_params = actor_out
            mean = dist_params[..., :action_dim]
            log_std = dist_params[..., action_dim:]
            log_std = jnp.tanh(log_std)
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
            std = jnp.exp(log_std)
            if approximate_mc:
                unsquashed_action = jax.random.normal(key_act, mean.shape) * std + mean
                y_t = jnp.tanh(unsquashed_action)
                action = (y_t * action_scale + action_bias)[0]
                unsquashed_action = unsquashed_action[0]
                log_prob = (
                    -((unsquashed_action - mean[0]) ** 2) / (2 * std[0] ** 2)
                    - 0.5 * jnp.log(2 * jnp.pi)
                    - jnp.log(std[0])
                    - jnp.log(
                        action_scale * (1 - jnp.tanh(unsquashed_action) ** 2) + 1e-6
                    )
                )
                prob = jnp.exp(log_prob.sum())
            else:
                unsquashed_action = jax.random.normal(key_act, mean.shape) * std + mean
                y_t = jnp.tanh(unsquashed_action)
                action = (y_t * action_scale + action_bias)[0]
            env_action = action
        _next_obs, new_env_state, reward, done, _ = env.step(
            key_step, env_state, env_action, env_params
        )
        # Reset carries when episode terminates (gymnax auto-resets the env state)
        new_actor_carry = jnp.where(
            done, jnp.zeros_like(new_actor_carry), new_actor_carry
        )
        new_cq1_carry = jnp.where(
            done, jnp.zeros_like(new_cq1_carry), new_cq1_carry
        )
        new_cq2_carry = jnp.where(
            done, jnp.zeros_like(new_cq2_carry), new_cq2_carry
        )
        if approximate_mc:
            new_mcq1_carry = jnp.where(
                done, jnp.zeros_like(new_mcq1_carry), new_mcq1_carry
            )
            new_mcq2_carry = jnp.where(
                done, jnp.zeros_like(new_mcq2_carry), new_mcq2_carry
            )
        transition = {
            obs_key: obs,
            action_key: jnp.atleast_1d(action),
            reward_key: jnp.asarray(reward, dtype=jnp.float32),
            terminated_key: jnp.asarray(done, dtype=jnp.float32),
            # Store pre-step carries — the state the networks were in *when
            # they produced this (obs, action)* — so burn-in can resume from
            # exactly here via params.burn_in_from_stored_carry.
            actor_carry_key: prev_actor_carry[0],
            critic_q1_carry_key: prev_cq1_carry[0],
            critic_q2_carry_key: prev_cq2_carry[0],
        }
        if approximate_mc:
            transition[behaviour_key] = prob
            transition[mc_critic_q1_carry_key] = prev_mcq1_carry[0]
            transition[mc_critic_q2_carry_key] = prev_mcq2_carry[0]
            if not is_discrete:
                transition[unsquashed_action_key] = jnp.atleast_1d(unsquashed_action)  # type: ignore
        new_online_buffer = online_buffer_functions.add(
            sac.online_buffer, transition, terminated=done
        )
        return (
            sac._replace(
                online_buffer=new_online_buffer,
                actor_online_carry=new_actor_carry,
                critic_q1_online_carry=new_cq1_carry,
                critic_q2_online_carry=new_cq2_carry,
                mc_critic_q1_online_carry=new_mcq1_carry if approximate_mc else sac.mc_critic_q1_online_carry,
                mc_critic_q2_online_carry=new_mcq2_carry if approximate_mc else sac.mc_critic_q2_online_carry,
            ),
            new_env_state,
        )

    def update_step_sac(sac: IQLearnState, key: jax.Array) -> Tuple[IQLearnState, dict]:
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
        # TODO: just lookup, remove
        if False:
            # Sequence (R2D2) training path
            key_sample, key_actor, key_critic, key_mc_critic = jax.random.split(key, 4)

            if is_discrete:

                def loss_actor_seq(actor):
                    _, outs = run_actor_scan(actor, learn_obs, _burn_ac)
                    probs = jax.nn.softmax(outs, axis=-1)
                    lp = jax.nn.log_softmax(outs, axis=-1)
                    _, _, qtw = run_critic_scan(
                        sac.critic_target,
                        learn_obs,
                        False,
                        jax.lax.stop_gradient(_burn_cq1),
                        jax.lax.stop_gradient(_burn_cq2),
                    )
                    qm = jnp.minimum(qtw[..., 0], qtw[..., 1])
                    ent = -(probs * lp).sum(-1)
                    v = (probs * qm).sum(-1) + sac.alpha * ent
                    L = -(v * learn_mask).sum() / jnp.maximum(learn_mask.sum(), 1.0)
                    return L, {
                        "q": (probs * qm).sum(-1).mean(),
                        "entropy": ent.mean(),
                        "v": v.mean(),
                    }

                def loss_critic_sac_seq(critic):
                    _, tg = run_actor_scan(sac.actor_target, learn_obs, _burn_ac)
                    tp = jax.nn.softmax(tg, axis=-1)
                    tlp = jax.nn.log_softmax(tg, axis=-1)
                    _, _, qtgt = run_critic_scan(
                        sac.critic_target, learn_obs, False, _burn_cq1, _burn_cq2
                    )
                    qtm = jnp.minimum(qtgt[..., 0], qtgt[..., 1])
                    vtgt = (tp * qtm).sum(-1) + sac.alpha * -(tp * tlp).sum(-1)
                    v_boot = (
                        _h_inv(vtgt, params.value_rescaling_eps)
                        if params.value_rescaling
                        else vtgt
                    )
                    G_n = _n_step_return_seq(learn_rew, learn_done, v_boot)
                    tq = jax.lax.stop_gradient(
                        _h(G_n, params.value_rescaling_eps)
                        if params.value_rescaling
                        else G_n
                    )
                    _, _, qon = run_critic_scan(
                        critic,
                        learn_obs,
                        False,
                        jax.lax.stop_gradient(_burn_cq1),
                        jax.lax.stop_gradient(_burn_cq2),
                    )
                    ai = jnp.round(learn_act.reshape(_bs, _TL)).astype(jnp.int32)
                    bi = jnp.arange(_bs)[:, None]
                    ti = jnp.arange(_TL)[None, :]
                    q1 = qon[bi, ti, ai, 0]
                    q2 = qon[bi, ti, ai, 1]
                    dn = jnp.maximum(learn_mask.sum(), 1.0)
                    L = 0.5 * (
                        ((q1 - tq) ** 2 * learn_mask).sum() / dn
                        + ((q2 - tq) ** 2 * learn_mask).sum() / dn
                    )
                    return L, {"critic_loss": L, "target_q": tq.mean()}

            else:

                def loss_actor_seq(actor):
                    _, outs = run_actor_scan(actor, learn_obs, _burn_ac)
                    m = outs[..., :action_dim]
                    ls = jnp.tanh(outs[..., action_dim:])
                    ls = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (ls + 1)
                    s = jnp.exp(ls)
                    u = jax.random.normal(key_actor, m.shape) * s + m
                    y = jnp.tanh(u)
                    a = y * action_scale + action_bias
                    lp = (
                        -((u - m) ** 2) / (2 * s**2)
                        - 0.5 * jnp.log(2 * jnp.pi)
                        - jnp.log(s)
                        - jnp.log(action_scale * (1 - y**2) + 1e-6)
                    ).sum(-1)
                    _, _, qtw = run_critic_scan(
                        sac.critic_target,
                        learn_obs,
                        a,
                        False,
                        jax.lax.stop_gradient(_burn_cq1),
                        jax.lax.stop_gradient(_burn_cq2),
                    )
                    qm = jnp.minimum(qtw[..., 0], qtw[..., 1])
                    v = qm - sac.alpha * lp
                    L = -(v * learn_mask).sum() / jnp.maximum(learn_mask.sum(), 1.0)
                    return L, {"q": qm.mean(), "entropy": -lp.mean(), "v": v.mean()}

                def loss_critic_sac_seq(critic):
                    _, tg = run_actor_scan(sac.actor_target, learn_obs, _burn_ac)
                    mt = tg[..., :action_dim]
                    lt = jnp.tanh(tg[..., action_dim:])
                    lt = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (lt + 1)
                    st = jnp.exp(lt)
                    ut = jax.random.normal(key_critic, mt.shape) * st + mt
                    yt = jnp.tanh(ut)
                    at = yt * action_scale + action_bias
                    lpt = (
                        -((ut - mt) ** 2) / (2 * st**2)
                        - 0.5 * jnp.log(2 * jnp.pi)
                        - jnp.log(st)
                        - jnp.log(action_scale * (1 - yt**2) + 1e-6)
                    ).sum(-1)
                    _, _, qtgt = run_critic_scan(
                        sac.critic_target, learn_obs, at, False, _burn_cq1, _burn_cq2
                    )
                    qtm = jnp.minimum(qtgt[..., 0], qtgt[..., 1])
                    vtgt = qtm - sac.alpha * lpt
                    v_boot = (
                        _h_inv(vtgt, params.value_rescaling_eps)
                        if params.value_rescaling
                        else vtgt
                    )
                    G_n = _n_step_return_seq(learn_rew, learn_done, v_boot)
                    tq = jax.lax.stop_gradient(
                        _h(G_n, params.value_rescaling_eps)
                        if params.value_rescaling
                        else G_n
                    )
                    _, _, qon = run_critic_scan(
                        critic,
                        learn_obs,
                        learn_act,
                        False,
                        jax.lax.stop_gradient(_burn_cq1),
                        jax.lax.stop_gradient(_burn_cq2),
                    )
                    q1 = qon[..., 0]
                    q2 = qon[..., 1]
                    dn = jnp.maximum(learn_mask.sum(), 1.0)
                    L = 0.5 * (
                        ((q1 - tq) ** 2 * learn_mask).sum() / dn
                        + ((q2 - tq) ** 2 * learn_mask).sum() / dn
                    )
                    return L, {"critic_loss": L, "target_q": tq.mean()}

            grads_actor, metrics = jax.grad(loss_actor_seq, has_aux=True)(sac.actor)
            grads_critic, metrics_critic = jax.grad(loss_critic_sac_seq, has_aux=True)(
                sac.critic
            )

            if approximate_mc:
                grads_mc_critic, metrics_mc_critic = jax.grad(
                    loss_mc_critic_sac, argnums=1, has_aux=True
                )(
                    sac.actor,
                    sac.mc_critic,
                    sac.mc_critic_target,
                    sac.online_buffer,
                    sac.alpha,
                    key_mc_critic,
                )

        else:
            key_sample, key_actor, key_critic, key_mc_critic = jax.random.split(key, 4)

            # Sample once; all three losses share this sequence + burned-in carries.
            shared_sample = sample_with_burn_in(
                sac.online_buffer,
                actor=sac.actor,
                critic=sac.critic,
                critic_target=sac.critic_target,
                key_sample=key_sample,
                mc_critic=sac.mc_critic if approximate_mc else None,
                mc_critic_target=sac.mc_critic_target if approximate_mc else None,
                actor_target=sac.actor_target,
            )

            _mc_for_disc = sac.mc_critic_target if approximate_mc else None

            # Actor gradient: maximise soft value V(s) = Q(s,a) - α log π(a|s)
            grads_actor, (metrics, refresh_actor) = jax.grad(
                loss_actor, has_aux=True
            )(
                sac.actor,
                sac.critic,
                shared_sample,
                sac.alpha,
                key_actor,
                _mc_for_disc,
            )
            # Critic gradient: minimise SAC Bellman MSE
            grads_critic, (metrics_critic, refresh_critic) = jax.grad(
                loss_critic_sac, argnums=1, has_aux=True
            )(
                sac.actor_target,
                sac.critic,
                sac.critic_target,
                shared_sample,
                sac.alpha,
                key_critic,
                _mc_for_disc,
            )

            if approximate_mc:
                grads_mc_critic, (metrics_mc_critic, refresh_mc) = jax.grad(
                    loss_mc_critic_sac, argnums=1, has_aux=True
                )(
                    sac.actor,
                    sac.mc_critic,
                    sac.mc_critic_target,
                    sac.online_buffer,
                    shared_sample,
                    sac.alpha,
                    key_mc_critic,
                    sac.critic_target,
                )
            else:
                refresh_mc = {}

        metrics.update(metrics_critic)

        if approximate_mc:
            metrics.update(metrics_mc_critic)

        # Diagnostic: pre-optimizer gradient magnitudes per network.  Useful
        # for detecting outlier updates that destabilise training.
        if params.diagnostics:
            def _grad_max_abs(grads):
                leaves = jax.tree_util.tree_leaves(grads)
                return jnp.max(jnp.stack([jnp.max(jnp.abs(l)) for l in leaves]))

            metrics["grad_norm_actor"] = optax.global_norm(grads_actor)
            metrics["grad_norm_critic"] = optax.global_norm(grads_critic)
            metrics["grad_max_abs_actor"] = _grad_max_abs(grads_actor)
            metrics["grad_max_abs_critic"] = _grad_max_abs(grads_critic)
            if approximate_mc:
                metrics["grad_norm_mc_critic"] = optax.global_norm(grads_mc_critic)
                metrics["grad_max_abs_mc_critic"] = _grad_max_abs(grads_mc_critic)

            # Diagnostic: online recurrent carry magnitudes — drift detection.
            metrics["carry_norm_actor"] = jnp.linalg.norm(
                sac.actor_online_carry.reshape(-1)
            )
            metrics["carry_norm_critic_q1"] = jnp.linalg.norm(
                sac.critic_q1_online_carry.reshape(-1)
            )
            if approximate_mc:
                metrics["carry_norm_mc_critic_q1"] = jnp.linalg.norm(
                    sac.mc_critic_q1_online_carry.reshape(-1)
                )

        # Carry refresh: scatter the per-step input carries that the loss
        # scans just produced into the online buffer at the sampled slots,
        # countering carry drift.  Each loss writes only its own carry key,
        # so dicts are disjoint and merge order is irrelevant.
        new_online_buffer_info = sac.online_buffer.info
        if params.refresh_stored_carries:
            new_online_buffer_info = dict(new_online_buffer_info)
            for refresh in (refresh_actor, refresh_critic, refresh_mc):
                for key, (idx, vals) in refresh.items():
                    new_online_buffer_info[key] = (
                        new_online_buffer_info[key].at[idx].set(vals)
                    )
        new_online_buffer = sac.online_buffer._replace(
            info=new_online_buffer_info
        )

        updates, new_actor_opt = actor_optimizer.update(
            grads_actor, sac.actor_optimizer_state
        )
        if params.diagnostics:
            metrics["update_norm_actor"] = optax.global_norm(updates)
        new_actor = optax.apply_updates(sac.actor, updates)  # type: ignore

        updates, new_critic_opt = critic_optimizer.update(
            grads_critic, sac.critic_optimizer_state
        )
        if params.diagnostics:
            metrics["update_norm_critic"] = optax.global_norm(updates)
        new_critic = optax.apply_updates(sac.critic, updates)  # type: ignore

        if approximate_mc:
            updates, new_mc_critic_opt = mc_critic_optimizer.update(
                grads_mc_critic, sac.mc_critic_optimizer_state
            )
            if params.diagnostics:
                metrics["update_norm_mc_critic"] = optax.global_norm(updates)
            new_mc_critic = optax.apply_updates(sac.mc_critic, updates)  # type: ignore

        if params.autotune_alpha:
            grads_alpha = jax.grad(loss_alpha)(sac.log_alpha, -metrics["entropy"])
            updates, new_alpha_opt = alpha_optimizer.update(
                grads_alpha, sac.alpha_optimizer_state
            )
            new_log_alpha = optax.apply_updates(sac.log_alpha, updates)  # type: ignore
            new_alpha = jnp.exp(new_log_alpha)  # type: ignore
        else:
            metrics.update({"alpha": sac.alpha})
            new_alpha_opt = sac.alpha_optimizer_state
            new_log_alpha = sac.log_alpha
            new_alpha = sac.alpha
        metrics.update({"alpha": new_alpha})

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
        if approximate_mc:
            new_mc_critic_target = jax.tree.map(
                lambda x, y: (1 - params.tau) * x + params.tau * y,
                sac.mc_critic_target,
                new_mc_critic,
            )

        if params.diagnostics:
            metrics["param_norm_actor"] = optax.global_norm(new_actor)
            metrics["param_norm_critic"] = optax.global_norm(new_critic)
            if approximate_mc:
                metrics["param_norm_mc_critic"] = optax.global_norm(new_mc_critic)

        return (
            IQLearnState(
                new_actor,  # type: ignore
                new_critic,  # type: ignore
                new_mc_critic if approximate_mc else None,
                new_actor_target,
                new_critic_target,
                new_mc_critic_target if approximate_mc else None,
                new_actor_opt,
                new_critic_opt,
                new_mc_critic_opt if approximate_mc else None,
                new_alpha_opt,  # type: ignore
                new_alpha,
                new_log_alpha,  # type: ignore
                new_online_buffer,
                sac.actor_online_carry,
                sac.critic_q1_online_carry,
                sac.critic_q2_online_carry,
                sac.mc_critic_q1_online_carry,
                sac.mc_critic_q2_online_carry,
            ),
            metrics,
        )

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

        # actor gradients (loss_actor returns aux=(metrics, refresh); IQ-Learn
        # path doesn't refresh stored carries, so refresh dict is discarded).
        grads_actor, (metrics, _refresh_actor) = jax.grad(
            loss_actor, has_aux=True
        )(
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
                iqlearn.mc_critic,
                new_actor_target,
                new_critic_target,
                iqlearn.mc_critic_target,
                new_actor_optimizer_state,
                new_critic_optimizer_state,
                iqlearn.mc_critic_optimizer_state,
                new_alpha_optimizer_state,  # type: ignore
                new_alpha,
                new_log_alpha,  # type: ignore
                iqlearn.online_buffer,
                iqlearn.actor_online_carry,
                iqlearn.critic_q1_online_carry,
                iqlearn.critic_q2_online_carry,
                iqlearn.mc_critic_q1_online_carry,
                iqlearn.mc_critic_q2_online_carry,
            ),
            metrics,
        )

    @partial(jax.jit, static_argnames=["env"])
    def _train_sac_jit(
        sac: IQLearnState,
        env,
        env_params,
        env_state,
        key: jax.Array,
    ) -> Tuple[IQLearnState, any, dict]:
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
        # Default reduction is mean over the round; for spike-detection
        # diagnostics also expose max-over-round under "{key}_max".
        max_keys = (
            "grad_norm_actor",
            "grad_norm_critic",
            "grad_norm_mc_critic",
            "update_norm_actor",
            "update_norm_critic",
            "update_norm_mc_critic",
            "sac_q_twin_gap_mean",
            "mc_td_residual_abs_mean",
        )
        round_max_keys = (
            "grad_max_abs_actor",
            "grad_max_abs_critic",
            "grad_max_abs_mc_critic",
            "target_q_max",
            "mc_target_q_max",
            "mc_pk_max",
            "disc_delta_abs_max",
        )
        max_extras = {
            f"{k}_round_max": metrics[k].max()
            for k in (max_keys + round_max_keys)
            if k in metrics
        }
        metrics = jax.tree.map(lambda x: x.mean(), metrics)
        metrics.update(max_extras)
        return sac, env_state, metrics

    def train_sac(
        sac: IQLearnState,
        env,
        env_params,
        env_state,
        key: jax.Array,
    ) -> Tuple[IQLearnState, any, dict]:
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
        return _train_sac_jit(sac, env, env_params, env_state, key)

    @jax.jit
    def train(iqlearn: IQLearnState, key: jax.Array) -> Tuple[IQLearnState, dict]:
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

    def prefill_buffer(
        sac: IQLearnState,
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
            if approximate_mc:
                transition[behaviour_key] = prob
                if not is_discrete:
                    transition[unsquashed_action_key] = jnp.atleast_1d(u)
            sac = sac._replace(
                online_buffer=online_buffer_functions.add(
                    sac.online_buffer, transition, terminated=terminated
                )
            )
        return sac, env_state

    graphs = IQLearnGraphs(
        actor=NetworkGraphs(actor_fe_graph, actor_memory_graph, actor_head_graph),
        critic_q1=NetworkGraphs(
            critic_q1_fe_graph, critic_q1_memory_graph, critic_q1_head_graph
        ),
        critic_q2=NetworkGraphs(
            critic_q2_fe_graph, critic_q2_memory_graph, critic_q2_head_graph
        ),
    )
    if is_discrete:
        def _public_get_importance_ratios(actor, obs, actions, behaviour_probs):
            batch = obs.shape[0]
            ratios, _ = get_importance_ratios(
                actor, obs, _make_actor_carry(batch), actions, behaviour_probs
            )
            return ratios
    else:
        def _public_get_importance_ratios(actor, obs, actions, behaviour_probs):
            ratios, _ = get_importance_ratios(actor, obs, actions, behaviour_probs)
            return ratios

    fns = IQLearnFunctions(
        predict, train, train_sac, _public_get_importance_ratios, prefill_buffer
    )
    if debug:
        # Wrap calculate_td_lambda to the historical debug test contract:
        # (actor, critic, online_buffer, indices) -> td (shape (len(indices),)).
        # Under the hood we build the per-index look-ahead windows and feed
        # zero carries (pre-carry-threading behaviour).
        N1 = params.lambda_truncation + 1
        _td_act_key = unsquashed_action_key if not is_discrete else action_key

        def _debug_calculate_td_lambda(actor, critic, online_buffer, indices):
            buf_size = online_buffer.info[obs_key].shape[0]
            td_idx = (indices[:, None] + jnp.arange(N1)[None, :]) % buf_size
            td_obs = online_buffer.info[obs_key][td_idx]
            td_act = online_buffer.info[_td_act_key][td_idx]
            td_beh = online_buffer.info[behaviour_key][td_idx]
            td_rew = online_buffer.info[reward_key][td_idx]
            td_done = online_buffer.info[terminated_key][td_idx]
            batch = indices.shape[0]
            a_c = _make_actor_carry(batch)
            q1c = (
                _make_mc_critic_q1_carry(batch)
                if approximate_mc
                else _make_critic_q1_carry(batch)
            )
            q2c = (
                _make_mc_critic_q2_carry(batch)
                if approximate_mc
                else _make_critic_q2_carry(batch)
            )
            td, _p_k, _fa, _fq1, _fq2 = calculate_td_lambda(
                actor,
                critic,
                iqlearn.alpha,
                jax.random.key(0),
                td_obs,
                td_act,
                td_beh,
                td_rew,
                td_done,
                a_c,
                q1c,
                q2c,
            )
            return td

        def _debug_get_q(critic, obs, actions, use_mc=False):
            """Debug wrapper: build zero carries and return just Q values."""
            batch = obs.shape[0]
            if use_mc and approximate_mc:
                q1c = _make_mc_critic_q1_carry(batch)
                q2c = _make_mc_critic_q2_carry(batch)
            else:
                q1c = _make_critic_q1_carry(batch)
                q2c = _make_critic_q2_carry(batch)
            q, _, _ = get_q(critic, obs, q1c, q2c, actions, use_mc)
            return q

        def _debug_get_entropy(actor, obs, key):
            batch = obs.shape[0]
            return get_entropy(actor, obs, _make_actor_carry(batch), key)

        return (
            iqlearn,
            fns,
            graphs,
            DebugFunctions(
                _debug_calculate_td_lambda,
                _debug_get_q,
                _debug_get_entropy,
                run_actor_scan,
                run_critic_scan,
            ),
        )
    return (iqlearn, fns, graphs)
