"""SAC actor-critic with optional λ-discrepancy V-trace critics and GVD.

Implements a SAC-style actor-critic agent.  When ``approximate_lambda=True``,
two additional V-trace λ-return critics (one per λ value) are trained
alongside the standard SAC critic; their Huber discrepancy is added to the
joint loss as a regulariser (the "λ-discrepancy" line of work).  When
``use_gvd=True``, two successor-feature V-heads are additionally trained on
observable-feature cumulants (``gvd_feature_fn(obs, prev_action)``) at
``gvd_lambda{1,2}`` and their squared discrepancy is added as a reward-free
representation regulariser (General Value Discrepancy; Koepernik et al.
2025).

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
unroll; ``jax.grad(..., argnums=[0,1,2,3,4,5,6])`` distributes gradients to
FE / actor / critic / λ1 / λ2 / SF1 / SF2.

Typical usage::

    key = jax.random.key(42)
    key_fe, key_heads = jax.random.split(key)
    fe = RecurrentFeatureExtractor(obs_shape, projection=128,
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

from .buffer import (
    Buffer,
    create_buffer,
    create_episode_aligned_sequence_sample,
    create_sequence_sample,
)

# Bounds for the squashed log-standard-deviation of the policy distribution.
# The raw output is tanh-squashed and then rescaled into this range to keep
# the distribution numerically stable while remaining expressive.
LOG_STD_MIN = -5
LOG_STD_MAX = 2


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
        layer_norm: If True, apply LayerNorm between each hidden Linear and
            its ReLU (RLPD-style).  No effect with ``hidden_dims=()``.
        rngs: Flax NNX RNG container used to initialise parameters.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        *,
        layer_norm: bool = False,
        rngs: nnx.Rngs,
    ):
        dims = [feature_dim] + list(hidden_dims) + [output_dim]
        self.layers = nnx.data(
            [nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)]
        )
        self.norms = nnx.data(
            [nnx.LayerNorm(d, rngs=rngs) for d in hidden_dims] if layer_norm else []
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Map a feature batch to the output space.

        Args:
            x: Feature batch of shape ``(batch, feature_dim)``.

        Returns:
            Output array of shape ``(batch, output_dim)``.
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.norms:
                x = self.norms[i](x)
            x = nnx.relu(x)
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
    agent was constructed with ``approximate_lambda=False``; the GVD
    successor-feature fields (``gvd_sf*``) are ``None`` when constructed
    with ``use_gvd=False``.

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
        gvd_sf1: Online successor-feature head for ``params.gvd_lambda1``
            (single V-style :class:`Head`, output dim = number of GVD
            features).  ``None`` when ``use_gvd=False``.
        gvd_sf2: Same, for ``params.gvd_lambda2``.
        gvd_sf1_target: EMA copy of ``gvd_sf1``.
        gvd_sf2_target: EMA copy of ``gvd_sf2``.
        gvd_sf1_optimizer_state: Optax state for the SF1 Adam optimiser.
        gvd_sf2_optimizer_state: Optax state for the SF2 Adam optimiser.
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
    update_step: jax.Array
    # GVD successor-feature heads (appended with defaults so positional
    # constructions and old pickles of the 19-field layout keep working).
    gvd_sf1: nnx.GraphState = None
    gvd_sf2: nnx.GraphState = None
    gvd_sf1_target: nnx.GraphState = None
    gvd_sf2_target: nnx.GraphState = None
    gvd_sf1_optimizer_state: optax.OptState = None
    gvd_sf2_optimizer_state: optax.OptState = None


class SACFunctions(NamedTuple):
    """Pure functions returned by :func:`create_iqlearn`.

    Attributes:
        predict: ``(state, obs, carry, key, deterministic, ...,
            prev_action=None) -> (action, new_carry[, extra])`` -- sample or
            compute a deterministic action for a single observation while
            threading the recurrent FE memory carry.  ``prev_action`` (optional,
            keyword) is the prev-action input for ``use_prev_action`` rollouts —
            ``None`` for the obs-only case; ``new_carry`` is the memory carry
            only (no action packed in).  Discrete: returns a ``float32`` action
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
            -> (state, env_state, env_carry, metrics)`` -- the un-jitted body
            behind :attr:`train`'s scan, with no host-side control flow.  Safe
            to ``jax.vmap`` over a leading seed axis to train many seeds
            concurrently in one kernel.  Unlike :attr:`train` it does **not**
            auto-prefill: the caller must warm the buffer first via
            :attr:`prefill_buffer` and thread ``env_carry`` explicitly.  The
            prev-action input is threaded internally and reset to zero per
            call (mirroring the per-call carry reset of :attr:`train`).
        encode_action: ``(action) -> encoding`` -- encode an executed action
            into the prev-action input expected by :attr:`predict`
            (``prev_action=``): a one-hot for discrete spaces, the squashed
            action values for continuous spaces.  Manual rollout loops that
            enable ``use_prev_action`` use this to thread the prev-action
            across steps; obs-only callers never need it.
    """

    predict: Callable
    train: Callable
    get_importance_ratios: Callable
    prefill_buffer: Callable
    train_unrolled: Callable
    encode_action: Callable


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
        calculate_latent: ``(fe_state, target_fe_state, observations,
            actions, dones, init_carries, init_prev_actions=None) ->
            (latent, target_latent)`` -- roll the online and target FEs over
            batch-major ``(B, T, ...)`` sequences with burn-in and per-step
            done resets (and, with ``use_prev_action``, the per-step
            prev-action input ``enc(a_{t-1})``).  Returns time-major latents
            of shape ``(T - burn_in_length, B, feature_dim)``.
        get_v: ``(actor, critic, graph, alpha, x, key, include_entropy,
            include_log, mask=None) -> values`` -- the policy-expected value
            ``V(s) = Σ_a π(a|s)·min(Q1,Q2)(s,a)`` (plus ``α·H(π)`` when
            ``include_entropy``).  This is the dense projection the
            ``dense_value_coef`` loss regresses; exposed for testing.
        graphs: The static ``nnx`` graphdefs the state pytrees pair with,
            keyed ``"critic"`` / ``"lambda1"`` / ``"lambda2"`` /
            ``"gvd_sf1"`` / ``"gvd_sf2"`` (absent keys = that head is
            disabled).  ``SACState`` carries only the *states*, so callers
            need these to evaluate a head outside the training loop.
        predict_qpi: ``(iqlearn, obs, carry, prev_action=None) ->
            (q_values, probs, new_carry)`` -- per-action Q (min-twin, raw/
            unmasked) and actor probabilities (masked softmax) for a single
            observation, plus the post-FE memory carry.  Discrete-only
            (``None`` for continuous action spaces).  Used for critic-Q /
            actor-π introspection; reads the existing critic/actor only.
    """

    get_q: Callable
    get_entropy: Callable
    calculate_latent: Callable
    predict_qpi: Callable | None = None
    get_v: Callable | None = None
    graphs: dict | None = None


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
            1.0 (i.e. assume on-policy data).  Used as an ablation.  Applies
            to the λ-critic and GVD successor-feature V-trace losses alike.
        gvd_coef: Multiplier on the GVD discrepancy regulariser
            ``mean((V_sf1 − V_sf2)²)``.  Ignored unless ``use_gvd=True``.
        gvd_lambda1: λ for the first GVD successor-feature head (paper
            default 0.0 — one-step TD).  Ignored unless ``use_gvd=True``.
        gvd_lambda2: λ for the second GVD successor-feature head (paper
            default 1.0 — Monte-Carlo).  Ignored unless ``use_gvd=True``.
        gvd_sf_lr: Learning rate for each SF-head Adam optimiser.  Ignored
            unless ``use_gvd=True``.
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
    periodic_update: int = 1
    lambda1: float = 0.1
    lambda2: float = 0.9
    c_bar: float = 1.0
    rho_bar: float = 1.0
    burn_in_length: int = 20
    sequence_length: int = 80
    lambda_truncation: int = 30
    lambda_coef: float = 0.2
    fake_onpolicy_loss: bool = True
    gvd_coef: float = 1.0
    gvd_lambda1: float = 0.0
    gvd_lambda2: float = 1.0
    gvd_sf_lr: float = 1e-4
    # If True, stop-gradient the online FE latents feeding the GVD SF heads so
    # neither the SF TD losses nor the discrepancy backprop into the shared FE
    # (a strictly stronger decoupling than gvd_coef=0, which leaves the SF
    # heads' own TD loss training the FE). The SF heads still learn.
    gvd_stop_fe: bool = False
    # If True, stop-gradient the FE latents feeding the ACTOR loss (SAC-AE
    # recipe): the shared FE is trained only by the value/critic (+ GVD) losses,
    # never by the policy-improvement gradient. The actor is a head on detached
    # features. Removes actor↔critic gradient conflict on the shared encoder.
    stop_actor_fe: bool = False
    # If True, stop-gradient the online FE latents feeding the CRITIC losses
    # (main twin critic + the λ-critics + their LD discrepancy). The value heads
    # then learn on detached features and never backprop into the shared FE.
    # Mirror of gvd_stop_fe for the SF heads: together the two knobs sever the
    # whole value side from the encoder, isolating whether value-magnitude
    # inflation is being driven into the shared recurrent memory by the critic.
    stop_critic_fe: bool = False
    # If True, replace the SAC E_π[Q] actor objective with an off-policy
    # V-trace (IMPALA-style) policy gradient.  The actor then maximises
    #   E[ ρ_t · Â_t · log π(a_t|s_t) ] + α·H(π),
    # where Â_t = ρ_t (r_t + γ(1-d_t) v_{t+1} − V(s_t)) is the V-trace
    # advantage: V(s) = Σ_a π(a|s) Q(s,a) from the twin-Q critic under the
    # target policy, and v_t the V-trace return (same backward recursion as the
    # λ-critics, full trace lam=1).  Only the taken action's log-prob carries a
    # gradient — there is no maximisation over the (mis-calibrated) untaken-action
    # Q, which is what drives the deadly-triad value-magnitude divergence under
    # value-greedy improvement.  The critic / λ-critic / (G)VD losses are
    # unchanged; the actor simply reads the critic as a baseline.  Discrete only.
    vtrace_actor: bool = False
    # If True, build the GVD successor-feature cumulant as the temporal
    # DIFFERENCE f_t = φ(o_{t+1}) − φ(o_t) (Jaderberg-style) instead of the raw
    # f_t = φ(o_t).  The paper uses this in its deep-RL section as an
    # SF-collapse remedy: a difference cumulant telescopes so the SF value
    # predicts (roughly) φ(o_end) − φ(o_t) — for a spatial hit map, the cells
    # that WILL be hit but haven't been = the remaining ship locations, which
    # is board-dependent and cannot be matched by a count-level marginal.
    # Removes the constant/marginal component that lets both λ-heads agree
    # trivially (the discrepancy → 0 without a Markov memory).  The difference
    # is zeroed across episode boundaries (via 1 − done at the terminal step).
    gvd_cumulant_diff: bool = False
    # If True, sample training windows that START at an episode start (the slot
    # after a terminal) instead of anywhere. The recurrent carry there is
    # genuinely zero, so the FE can be rolled from a zero carry with NO burn-in
    # and NO stored carry — eliminating stored-carry staleness (drift-free by
    # construction). Requires sequence_length >= the longest episode to contain
    # a whole episode per window. Pair with burn_in_from_stored_carry=False and
    # burn_in_length=0.
    episode_aligned_sampling: bool = False
    # If True, subtract the batch-mean advantage in the V-trace actor loss
    # before the policy-gradient term (standard A2C/IMPALA variance reduction).
    # Targets the observed systematic advantage offset (lagging value baseline →
    # almost all advantages positive → policy reinforces sampled spread and
    # never sharpens). Only affects the vtrace_actor path.
    vtrace_center_advantage: bool = False
    # Full PPO-style advantage standardisation (Â − mean)/std in the V-trace
    # actor loss. Takes precedence over vtrace_center_advantage: centering alone
    # leaves the advantages at a tiny absolute scale under low-variance returns,
    # so the PG gradient is negligible and the policy freezes at ~uniform.
    vtrace_normalize_advantage: bool = False
    # PPO clipped-surrogate epsilon for the V-trace actor (0 = plain PG, no trust
    # region).  LOAD-BEARING: without it the unclipped off-policy PG drives one
    # logit to +∞, entropy collapses to ~0 within a few steps, the policy becomes
    # a deterministic board-blind firing order, and the encoder's board memory
    # dies with it (forward ladder: 0.865 clipped vs 0.591 unclipped).
    ppo_clip_eps: float = 0.0
    # If True, mask out every step after the window's first episode in the
    # sequence losses.  With episode_aligned_sampling a short episode's window
    # spills into the NEXT episode's first steps, so those early/low-memory
    # states are ~2x over-represented (once as their own window's start, once as
    # the previous window's tail) and the FE gradient is biased AWAY from the
    # memory-rich late states.  Reference PPO's arbitrary-start chunks sample
    # positions uniformly and have no such bias.
    mask_first_episode_only: bool = False
    # ---- dense (expected) value loss -------------------------------------
    # THE FIX for off-policy memory formation (forward ladder, ablations.md
    # Part J).  A per-action (twin-)Q head regressed ONLY at the executed action
    # trains 1 of `action_dim` output columns per step, so the gradient reaching
    # the shared recurrent encoder is ~action_dim times sparser than a scalar-V
    # head's — and WHICH column receives it is dictated by the behaviour policy's
    # samples.  Under a near-uniform off-policy actor those updates scatter into
    # noise and the encoder never accumulates the board.  Adding a DENSE term
    # that regresses V(s) = Σ_a π(a|s)·Q(s,a) (an expected-SARSA-style
    # projection) against the SAME V-trace target restores a dense encoder
    # gradient: in the ladder this recovered fired_auroc 0.708 -> 0.969 with no
    # change to the actor.  0.0 = off (prior behaviour).
    dense_value_coef: float = 0.0
    # If False, drop the taken-action regression and keep ONLY the dense term
    # (the exact form validated in the ladder).  Safe when the policy is a
    # separate head (vtrace_actor); with SAC-style extraction the per-action Q
    # would lose its action discrimination, so the default keeps both.
    sparse_value_loss: bool = True
    # If True, compute the λ-discrepancy (and the GVD discrepancy) between the
    # heads' EXPECTED values V(s) = Σ_a π(a|s)·Q(s,a) instead of between
    # Q(s,a_t) at the executed action.  Same sparsity argument as
    # dense_value_coef — and it is what the reference actually does: its
    # λ-discrepancy term is on the scalar V heads, not on per-action Q.
    dense_discrepancy: bool = False
    # Weight on the main SAC critic's 1-step Bellman loss in the joint loss.
    # V(s) ≈ r + γV(s') is satisfiable by a near-CONSTANT value with no history,
    # so this always-on head is the most memoryless-friendly pressure on the
    # shared recurrent encoder — and the reference has no equivalent (its value
    # is trained on λ=0.95 near-Monte-Carlo returns, which encode
    # steps-remaining and therefore require integrating past hits).
    # 0.0 drops it, leaving the FE shaped only by the λ-critics (+ discrepancy).
    sac_critic_coef: float = 1.0
    # Scalar multiplier on the GVD successor-feature cumulant. The SF value is
    # E[Σ γ^t c_t]; for a raw hit cumulant (c∈{0,1}) that grows to ~O(1/(1-γ)),
    # whose large TD targets drive the SF heads into deadly-triad divergence
    # (sf1 → large negative, violating the c≥0 lower bound → garbage discrepancy).
    # Setting scale≈(1-γ) turns the SF into a bounded "expected future hit-rate"
    # in [0,1], stabilising the regression while preserving the (interpretable)
    # signal. The discrepancy scales by scale², so retune gvd_coef accordingly.
    gvd_cumulant_scale: float = 1.0
    # If True, the online behaviour policy executes a uniform-random LEGAL action
    # each step (instead of the actor's), giving board-revealing, order-invariant
    # data with no learned-policy structure. Pair with actor_lr=0/critic_lr=0 +
    # stop_actor_fe + stop_critic_fe to train the FE by the GVD objective ALONE
    # on random data — the clean isolation of reward-free memory-forcing.
    random_behaviour: bool = False
    # Global gradient-norm clip applied to every optimizer (FE / actor / critic /
    # λ-critics / GVD-SF heads), matching the reference PPO's MAX_GRAD_NORM=0.5.
    # 0.0 disables clipping (the prior behaviour). Unclipped gradients on the
    # large-magnitude (+100) reward + off-policy value learning are a primary
    # deadly-triad divergence driver — set ~0.5 to stabilise.
    grad_clip: float = 0.0


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


def sf_vtrace_targets(
    V: jax.Array,
    cumulants: jax.Array,
    dones: jax.Array,
    ratios: jax.Array,
    gamma: float,
    lam: float,
    rho_bar: float,
    c_bar: float,
) -> jax.Array:
    """Vector V-trace λ-return targets for a successor-feature head (GVD).

    Componentwise-over-features version of the scalar backward recursion in
    ``loss_vtrace_lambda_sequence`` (see ``scan_target`` there): a reverse
    ``jax.lax.scan`` over time computing, per step ``t`` and feature ``i``,

        rho_t   = min(rho_bar, ratio_t)                       # (B, 1) bcast
        c_t     = lam * min(c_bar, ratio_t)
        delta_t = f_t + (1 - done_t) * gamma * V_{t+1} - V_t
        v_t     = V_t + rho_t * delta_t
                  + (1 - done_t) * gamma * c_t * (v_{t+1} - V_{t+1})

    with zero terminal carry ``(v_{T'}, V_{T'}) = (0, 0)`` — the missing
    bootstrap mass at the unroll tail is absorbed by the caller dropping the
    last ``lambda_truncation`` steps, exactly as in the scalar version.
    Ratios are per-step scalars shared across features (broadcast via
    ``[..., None]``).

    Args:
        V: ``(T', B, n)`` state values ``V(s_t)`` from the TARGET SF head.
        cumulants: ``(T', B, n)`` per-step feature pseudo-rewards ``f_t``
            (e.g. ``φ(o_{t+1}) − φ(o_t)``, zeroed across episode boundaries).
        dones: ``(T', B)`` termination flags (float 0/1).
        ratios: ``(T', B)`` importance ratios ``π(a|s) / b(a|s)``.
        gamma: Discount factor.
        lam: λ value for this head (``params.gvd_lambda{1,2}``).
        rho_bar: V-trace ρ truncation cap.
        c_bar: V-trace c truncation cap.

    Returns:
        ``(T', B, n)`` targets ``v_t``; the caller must ``stop_gradient``
        them and drop the trailing ``lambda_truncation`` steps before the
        regression loss.
    """

    def scan_target(scan_carry, x):
        v_sp1, V_sp1 = scan_carry
        V_s, done, f, ratio = x
        # ratios / dones are per-step scalars shared across the n features.
        rho = jnp.minimum(rho_bar, ratio)[..., None]
        c = (lam * jnp.minimum(c_bar, ratio))[..., None]
        nd = (1.0 - done)[..., None]
        delta_V = f + nd * gamma * V_sp1 - V_s
        v_s = V_s + rho * delta_V + nd * gamma * c * (v_sp1 - V_sp1)
        return (v_s, V_s), v_s

    _carry, targets = jax.lax.scan(
        scan_target,
        (jnp.zeros_like(V[0]), jnp.zeros_like(V[0])),
        (V, dones, cumulants, ratios),
        reverse=True,
        unroll=8,  # fuse across timesteps -> fewer tiny per-step kernels
    )
    return targets


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

unsquashed_action_key = "unsquashed_actions"
reward_key = "rewards"
terminated_key = "terminated"
behaviour_key = "behaviour_weight"
carry_key = "carries"
prev_action_key = "prev_actions"


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
    use_prev_action: bool = False,
    critic_layer_norm: bool = False,
    obs_fn: Callable[[jax.Array], jax.Array] = lambda obs: obs,
    mask_fn: Callable[[jax.Array], jax.Array] | None = None,
    burn_in_from_stored_carry: bool = False,
    use_gvd: bool = False,
    gvd_feature_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    gvd_sf_dims: tuple[int, ...] = (128,),
    debug: bool = False,
    use_sac: bool = True,
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
            ``feature_extractor(carry, obs, prev_action=None) -> (new_carry,
            y)`` calling convention and an ``initialize_carry(batch_size)``
            helper (see ``utils.RecurrentFeatureExtractor``).  Ownership is
            transferred;
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
        use_prev_action: If True, the previously executed action is fed to
            the feature extractor's projection alongside the observation
            (encoded as a one-hot for discrete, squashed values for
            continuous).  The FE must have been built with
            ``prev_action_dim == action_dim``.  The prev-action input is
            threaded *next to* the carry (not packed into it): ``run_env_step``
            feeds the previously executed action back into ``predict`` and the
            training scans thread ``enc(a_{t-1})`` per step, both reset to zero
            at episode boundaries (= "no previous action").  Manual rollout
            loops compute the next prev-action from the executed action via
            ``fns.encode_action`` (so executing a different action than
            ``predict`` returned is handled by encoding whatever was executed).
        critic_layer_norm: If True, apply LayerNorm before the ReLU in every
            hidden layer of all critic heads (SAC twin + λ-critics; the actor
            is untouched).  Stabilises the one-step bootstrap against
            representation drift in the shared FE — without it, the
            λ-discrepancy term (which trains the FE only, since the λ-heads
            are frozen in ``loss_ld``) can drive unbounded Q growth.
        obs_fn: Pure function mapping the raw stored observation to the
            observation actually fed to the feature extractor.  Defaults to
            the identity.  Use it to hide part of the observation from the
            network (e.g. an action mask packed into the obs); the *full* raw
            observation is still stored in the buffer so ``mask_fn`` can
            recover the masked view during the loss.  The feature extractor
            must already be sized for ``obs_fn``'s output.
        mask_fn: Optional pure function mapping the raw stored observation to
            a boolean (or 0/1) action mask of shape ``(*batch, action_dim)``.
            When provided (discrete only), invalid actions are removed from
            the policy everywhere logits become a categorical distribution
            (action selection, soft value, entropy, importance ratios, and the
            random pre-fill policy).  ``None`` (default) disables masking and
            leaves behaviour identical to before.
        burn_in_from_stored_carry: If True, store the memory carry (and, with
            ``use_prev_action``, the prev-action input ``enc(a_{t-1})`` — not
            reconstructable inside a sampled window) that the online policy
            consumed at every step, and initialise the burn-in of each sampled
            training sequence from the stored carry / prev-action of its first
            step instead of zeros (R2D2's "stored state" strategy).  Stored carries are stale with respect to the
            current FE weights — the burn-in refreshes them — but typically
            allow a much shorter ``params.burn_in_length``.  Costs
            ``carry_dim * online_buffer_size * 4`` bytes of extra buffer
            memory.  Pre-fill transitions keep zero carries (the random
            pre-fill policy threads no memory); zeros mean "episode start",
            which the burn-in absorbs.
        use_gvd: If True, additionally train two successor-feature (SF)
            V-heads via V-trace λ-returns over feature pseudo-rewards
            (``gvd_lambda{1,2}``) and add their squared discrepancy to the
            joint loss, weighted by ``params.gvd_coef`` (General Value
            Discrepancy; Koepernik et al. 2025).  Like ``loss_ld``, the
            discrepancy gradient reaches only the shared FE — the SF heads
            are frozen inside the discrepancy term.  Discrete-only.
        gvd_feature_fn: Pure function ``(obs, prev_action) -> features`` mapping
            the **raw stored observation** ``o_t`` (before ``obs_fn`` — same
            convention as ``mask_fn``) together with the **previous action**
            ``a_{t-1}`` (float32 index, shape ``(*batch, 1)``) to a feature
            vector ``(*batch, n_features)``.  ``φ(o_t, a_{t-1})`` is used
            directly as the per-step SF cumulant (paper pseudo-return
            ``Σ γ^t f(ω_t)``).  Passing ``a_{t-1}`` lets the cumulant localise
            the per-step outcome ``o_t[hit_bit]`` (which is the result of
            ``a_{t-1}``) to the cell that was fired — e.g. a spatial hit map —
            so the SF discrepancy can pressure the FE to remember *where* past
            hits occurred, not just aggregate counts.  ``a_{t-1}`` is zeroed at
            the window start and across ``done`` resets; callers must gate any
            action-localised term by the hit bit (which is 0 at episode starts)
            so those placeholder actions contribute nothing.  Required when
            ``use_gvd=True``.
        gvd_sf_dims: Hidden layer widths for each SF head.
        debug: If True, additionally return a :class:`DebugFunctions` named
            tuple exposing internal helpers (``get_q``, ``get_entropy``).

    Returns:
        When ``debug=False`` (default): a ``(SACState, SACFunctions)`` pair.
        When ``debug=True``: a ``(SACState, SACFunctions, DebugFunctions)``
        triple.  ``SACState`` is the initial agent state with online and
        target networks set to the same weights; ``SACFunctions`` bundles
        ``predict`` / ``train`` / ``get_importance_ratios`` / ``prefill_buffer``.
    """
    fe_prev_action_dim = getattr(feature_extractor, "prev_action_dim", 0)
    if use_prev_action and fe_prev_action_dim != action_dim:
        raise ValueError(
            f"use_prev_action=True requires a feature extractor built with "
            f"prev_action_dim == action_dim ({action_dim}); "
            f"got prev_action_dim={fe_prev_action_dim}."
        )
    if not use_prev_action and fe_prev_action_dim != 0:
        raise ValueError(
            f"feature extractor was built with prev_action_dim="
            f"{fe_prev_action_dim} but use_prev_action=False; "
            f"pass use_prev_action=True or rebuild the FE with "
            f"prev_action_dim=0."
        )
    if use_gvd:
        if gvd_feature_fn is None:
            raise ValueError("use_gvd=True requires gvd_feature_fn.")
        if not is_discrete:
            raise ValueError(
                "use_gvd=True is only supported for discrete action spaces "
                "(the SF V-trace importance ratios use the discrete path)."
            )
        if params.lambda_truncation < 1:
            raise ValueError(
                "use_gvd=True requires params.lambda_truncation >= 1 (the "
                "zero-padded final SF cumulant must fall in the truncated "
                "tail)."
            )

    if is_discrete:
        def encode_action(action: jax.Array) -> jax.Array:
            # Stored actions are float32 indices, shape (..., 1) or scalar.
            idx = jnp.round(action).astype(jnp.int32)
            if idx.ndim and idx.shape[-1] == 1:
                idx = idx[..., 0]
            return jax.nn.one_hot(idx, action_dim, dtype=jnp.float32)
    else:
        def encode_action(action: jax.Array) -> jax.Array:
            # The executed (squashed) env action, as in the original
            # ActionConcatWrapper — not the unsquashed pre-tanh value.
            return action.reshape(*action.shape[:-1], action_dim)

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
    if burn_in_from_stored_carry:
        # Memory carry consumed by the policy at each step; used to initialise
        # the training burn-in.
        online_shapes[carry_key] = (feature_extractor.carry_dim,)
        if use_prev_action:
            # The prev-action encoding consumed at each step (now threaded
            # alongside the carry rather than packed into it); used to seed the
            # burn-in's first-step prev-action input.
            online_shapes[prev_action_key] = (action_dim,)
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
    if burn_in_from_stored_carry:
        online_mc_this_keys.append(carry_key)
        if use_prev_action:
            online_mc_this_keys.append(prev_action_key)
    if params.episode_aligned_sampling:
        # Windows begin at episode starts (previous slot terminal) → the carry
        # there is genuinely zero, so training rolls the FE from zero with no
        # burn-in / no stored carry: drift-free, no staleness. Pair with
        # burn_in_from_stored_carry=False and (ideally) burn_in_length=0.
        online_buffer_lambda_sample = create_episode_aligned_sequence_sample(
            online_buffer.size,
            params.online_batch_size,
            params.burn_in_length + params.sequence_length + params.lambda_truncation,
            online_mc_this_keys,
            terminated_key,
        )
    else:
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
    # FE call returns (new_carry, y); we only need y's last dim.  ``obs_fn`` is
    # applied first so the dummy matches the (possibly reduced) observation the
    # FE was sized for.
    dummy_obs = obs_fn(jnp.zeros((1,) + buffer.info[obs_key].shape[1:]))
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
            rngs=nnx.Rngs(k_actor),
        )
        critic_q1_model = Head(
            feature_dim,
            critic_dims,
            action_dim,
            layer_norm=critic_layer_norm,
            rngs=nnx.Rngs(k_cq1),
        )
        critic_q2_model = Head(
            feature_dim,
            critic_dims,
            action_dim,
            layer_norm=critic_layer_norm,
            rngs=nnx.Rngs(k_cq2),
        )
        if approximate_lambda:
            lambda1_q1_critic_model = Head(
                feature_dim,
                lambda1_critic_dims,
                action_dim,
                layer_norm=critic_layer_norm,
                rngs=nnx.Rngs(k_l1q1),
            )
            lambda1_q2_critic_model = Head(
                feature_dim,
                lambda1_critic_dims,
                action_dim,
                layer_norm=critic_layer_norm,
                rngs=nnx.Rngs(k_l1q2),
            )
            lambda2_q1_critic_model = Head(
                feature_dim,
                lambda2_critic_dims,
                action_dim,
                layer_norm=critic_layer_norm,
                rngs=nnx.Rngs(k_l2q1),
            )
            lambda2_q2_critic_model = Head(
                feature_dim,
                lambda2_critic_dims,
                action_dim,
                layer_norm=critic_layer_norm,
                rngs=nnx.Rngs(k_l2q2),
            )
        if use_gvd:
            # Infer the feature width from a dummy call on the RAW obs shape
            # (gvd_feature_fn sees the full stored observation, like mask_fn)
            # plus a dummy prev-action index.
            dummy_phi = gvd_feature_fn(
                jnp.zeros((1,) + buffer.info[obs_key].shape[1:]),
                jnp.zeros((1, 1), dtype=jnp.float32),
            )
            n_gvd_features = dummy_phi.shape[-1]
            # fold_in keeps the existing split(key, 7) stream untouched so
            # use_gvd=False initialisation stays bit-identical to before.
            k_sf1, k_sf2 = jax.random.split(jax.random.fold_in(key, 0x6FD), 2)
            # Q-style per-action SF heads: one n_gvd_features-vector per
            # action, flat output reshaped to (batch, action_dim, n) in
            # run_sf (mirrors the all-actions layout of the λ-critics).
            gvd_sf1_model = Head(
                feature_dim,
                gvd_sf_dims,
                action_dim * n_gvd_features,
                layer_norm=critic_layer_norm,
                rngs=nnx.Rngs(k_sf1),
            )
            gvd_sf2_model = Head(
                feature_dim,
                gvd_sf_dims,
                action_dim * n_gvd_features,
                layer_norm=critic_layer_norm,
                rngs=nnx.Rngs(k_sf2),
            )
    else:
        actor_model = Head(
            feature_dim,
            actor_dims,
            2 * action_dim,
            rngs=nnx.Rngs(k_actor),
        )
        # For continuous critics, features and actions are concatenated before
        # the head, so input_dim = feature_dim + action_dim, output_dim = 1.
        critic_q1_model = Head(
            feature_dim + action_dim,
            critic_dims,
            1,
            layer_norm=critic_layer_norm,
            rngs=nnx.Rngs(k_cq1),
        )
        critic_q2_model = Head(
            feature_dim + action_dim,
            critic_dims,
            1,
            layer_norm=critic_layer_norm,
            rngs=nnx.Rngs(k_cq2),
        )
        if approximate_lambda:
            lambda1_q1_critic_model = Head(
                feature_dim + action_dim,
                lambda1_critic_dims,
                1,
                layer_norm=critic_layer_norm,
                rngs=nnx.Rngs(k_l1q1),
            )
            lambda1_q2_critic_model = Head(
                feature_dim + action_dim,
                lambda1_critic_dims,
                1,
                layer_norm=critic_layer_norm,
                rngs=nnx.Rngs(k_l1q2),
            )
            lambda2_q1_critic_model = Head(
                feature_dim + action_dim,
                lambda2_critic_dims,
                1,
                layer_norm=critic_layer_norm,
                rngs=nnx.Rngs(k_l2q1),
            )
            lambda2_q2_critic_model = Head(
                feature_dim + action_dim,
                lambda2_critic_dims,
                1,
                layer_norm=critic_layer_norm,
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
    if use_gvd:
        gvd_sf1_graph, gvd_sf1_state = nnx.split(gvd_sf1_model)
        gvd_sf2_graph, gvd_sf2_state = nnx.split(gvd_sf2_model)

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

    # Optimizers: actor operates on its nnx.GraphState; critic on TwinCriticState.
    def _mk_opt(lr):
        # Optionally clip the global gradient norm (reference MAX_GRAD_NORM=0.5)
        # before Adam. grad_clip <= 0 disables (prior behaviour).
        if params.grad_clip and params.grad_clip > 0:
            return optax.chain(optax.clip_by_global_norm(params.grad_clip),
                               optax.adam(lr))
        return optax.adam(lr)

    fe_optimizer = _mk_opt(params.fe_lr)
    actor_optimizer = _mk_opt(params.actor_lr)
    critic_optimizer = _mk_opt(params.critic_lr)
    if approximate_lambda:
        lambda1_critic_optimizer = _mk_opt(params.lambda_critic_lr)
        lambda2_critic_optimizer = _mk_opt(params.lambda_critic_lr)
    if use_gvd:
        gvd_sf1_optimizer = _mk_opt(params.gvd_sf_lr)
        gvd_sf2_optimizer = _mk_opt(params.gvd_sf_lr)
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
    if use_gvd:
        gvd_sf1_optimizer_state = gvd_sf1_optimizer.init(gvd_sf1_state)
        gvd_sf2_optimizer_state = gvd_sf2_optimizer.init(gvd_sf2_state)
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
        jnp.zeros(tuple(), dtype=jnp.int32),
        gvd_sf1=remove_weak_types(gvd_sf1_state) if use_gvd else None,
        gvd_sf2=remove_weak_types(gvd_sf2_state) if use_gvd else None,
        # SF targets start equal to online weights, like all other targets.
        gvd_sf1_target=remove_weak_types(gvd_sf1_state) if use_gvd else None,
        gvd_sf2_target=remove_weak_types(gvd_sf2_state) if use_gvd else None,
        gvd_sf1_optimizer_state=(
            remove_weak_types(gvd_sf1_optimizer_state) if use_gvd else None
        ),
        gvd_sf2_optimizer_state=(
            remove_weak_types(gvd_sf2_optimizer_state) if use_gvd else None
        ),
    )

    # ------------------------------------------------------------------
    # Shared helper: actor forward pass (same for both action space types)
    # ------------------------------------------------------------------

    def run_actor(actor: nnx.GraphState, x: jax.Array) -> jax.Array:
        """Reconstruct and run the actor head on pre-extracted features x."""
        head = nnx.merge(actor_graph, actor)
        return head(x)

    def _apply_mask(logits: jax.Array, mask: jax.Array | None) -> jax.Array:
        """Drive logits of illegal actions to a large finite negative value.

        ``mask`` is a boolean / 0-1 array broadcastable to ``logits``
        (``(*batch, num_actions)``); ``None`` returns the logits unchanged.
        Rows with no legal action (e.g. zero-padded burn-in observations) are
        left fully unmasked so the subsequent ``softmax`` is well defined.

        A finite fill (``-1e9``) is used rather than ``-inf`` on purpose: with
        ``-inf`` the masked ``log_softmax`` entries become ``-inf`` and the
        ``probs * log_probs`` entropy term has a ``0 * -inf`` gradient that
        back-propagates as ``NaN`` (the ``jnp.where``-with-``inf`` trap),
        contaminating the shared feature extractor.  ``-1e9`` makes the masked
        probabilities underflow to exactly ``0`` while keeping all gradients
        finite.
        """
        if mask is None:
            return logits
        mask = mask.astype(bool)
        safe = jnp.any(mask, axis=-1, keepdims=True)
        keep = jnp.logical_or(mask, jnp.logical_not(safe))
        return jnp.where(keep, logits, -1e9)

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
            mask: jax.Array | None = None,
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
                mask: Optional action mask ``(batch, num_actions)``; illegal
                    actions are dropped from the policy before the softmax.

            Returns:
                When ``include_log=False``: value array of shape ``(batch,)``.
                When ``include_log=True``: ``(values, metrics_dict)``.
            """
            logits = _apply_mask(run_actor(actor, x), mask)  # (batch, num_actions)
            probs = jax.nn.softmax(logits)  # (batch, num_actions)
            log_probs = jax.nn.log_softmax(logits)  # (batch, num_actions)
            q_twin = run_critic(critic, graph, x)  # (batch, num_actions, 2)
            q_min = jnp.minimum(q_twin[..., 0], q_twin[..., 1])  # (batch, num_actions)
            # Illegal actions have probs≈0 and a large-but-finite log_probs, so
            # their contribution to the entropy is ~0 with finite gradients.
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

        def get_entropy(actor, x, _key, mask=None):
            logits = _apply_mask(run_actor(actor, x), mask)  # (batch, num_actions)
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
            prev_action: jax.Array | None = None,
        ) -> jax.Array | Tuple[jax.Array, jax.Array]:
            """Compute a discrete action for a single observation.

            Args:
                iqlearn: Current agent state.
                obs: Single observation of shape ``(*obs_shape,)`` (no batch
                    dim).
                carry: Recurrent FE memory carry of shape ``(carry_dim,)`` from
                    the previous step (zero at episode start).
                prev_action: Encoding of the previously executed action, shape
                    ``(action_dim,)`` (zeros = episode start).  Optional —
                    ``None`` (the obs-only / ``use_prev_action=False`` case) is
                    treated as no previous action.  Compute the next step's
                    value from the executed action via ``encode_action``.
                key: JAX PRNG key, used only when ``deterministic=False``.
                deterministic: If True, return the greedy (argmax) action.
                    If False, sample from the categorical policy.
                return_prob: If True, also return ``π(a|s)`` for the selected
                    action.

            Returns:
                ``(action, new_carry)``, or ``(action, new_carry, prob)`` when
                ``return_prob=True``.  ``action`` is a ``float32`` index;
                ``new_carry`` is the memory carry only.
            """
            obs_batch = jnp.expand_dims(obs_fn(obs), 0)
            carry_batch = jnp.expand_dims(carry, 0)
            prev_action_batch = (
                jnp.expand_dims(prev_action, 0) if prev_action is not None else None
            )
            new_carry, x = nnx.merge(
                feature_extractor_graph, iqlearn.feature_extractor
            )(carry_batch, obs_batch, prev_action_batch)
            mask = mask_fn(obs) if mask_fn is not None else None
            logits = _apply_mask(run_actor(iqlearn.actor, x)[0], mask)  # (num_actions,)
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
        def predict_qpi(
            iqlearn: SACState,
            obs: jax.Array,
            carry: jax.Array,
            prev_action: jax.Array | None = None,
        ) -> Tuple[jax.Array, jax.Array, jax.Array]:
            """Diagnostic: per-action Q (min-twin) and actor probabilities for
            a single state, plus the post-FE carry.

            Returns ``(q_values, probs, new_carry)`` where ``q_values`` and
            ``probs`` are ``(num_actions,)`` (raw Q — NOT masked; probs are the
            masked-softmax policy).  ``new_carry`` is the memory carry only —
            the prev-action input is threaded separately (``prev_action``,
            mirroring ``predict``); pass the previously executed action's
            encoding to feed it into the FE (``None`` = zeros / episode start).
            Used by the critic-Q / actor-π heatmaps; mirrors ``predict``'s FE
            roll exactly and reads the existing critic/actor only.
            """
            obs_batch = jnp.expand_dims(obs_fn(obs), 0)
            carry_batch = jnp.expand_dims(carry, 0)
            prev_action_batch = (
                jnp.expand_dims(prev_action, 0) if prev_action is not None else None
            )
            new_carry, x = nnx.merge(
                feature_extractor_graph, iqlearn.feature_extractor
            )(carry_batch, obs_batch, prev_action_batch)
            mask = mask_fn(obs) if mask_fn is not None else None
            logits = _apply_mask(run_actor(iqlearn.actor, x)[0], mask)
            probs = jax.nn.softmax(logits)               # (num_actions,)
            q_twin = run_critic(iqlearn.critic, critic_graph, x)  # (1, n, 2)
            q_min = jnp.minimum(q_twin[..., 0], q_twin[..., 1])[0]  # (num_actions,)
            return q_min, probs, new_carry[0]

        @jax.jit
        def get_importance_ratios(
            actor: nnx.GraphState,
            x: jax.Array,
            actions: jax.Array,
            behaviour_probs: jax.Array,
            mask: jax.Array | None = None,
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
                mask: Optional action mask broadcastable to the logits;
                    illegal actions are dropped before the softmax.

            Returns:
                Importance ratios of shape ``(*batch_dims,)``.
            """
            logits = _apply_mask(run_actor(actor, x), mask)  # (*batch_dims, num_actions)
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
            prev_action: jax.Array | None = None,
        ) -> jax.Array | Tuple[jax.Array, jax.Array]:
            """Compute a continuous action for a single observation.

            Args:
                iqlearn: Current agent state.
                obs: Single observation of shape ``(*obs_shape,)`` (no batch
                    dim).
                carry: Recurrent FE memory carry of shape ``(carry_dim,)`` from
                    the previous step (zero at episode start).
                key: JAX PRNG key, used only when ``deterministic=False``.
                deterministic: If True, return the tanh-squashed policy mean
                    (no sampling noise).  If False, sample from the full
                    Gaussian policy.
                return_unsquashed: If True, also return the pre-tanh action
                    (needed for stored behaviour probabilities).
                prev_action: Encoding of the previously executed action, shape
                    ``(action_dim,)`` (zeros = episode start).  Optional —
                    ``None`` (the obs-only / ``use_prev_action=False`` case) is
                    treated as no previous action.  For continuous spaces the
                    encoding is the executed (squashed) action; compute the
                    next step's value via ``encode_action``.

            Returns:
                ``(action, new_carry)``, or
                ``(action, new_carry, unsquashed_action)`` when
                ``return_unsquashed=True``.  ``action`` has shape
                ``(action_dim,)`` and is scaled / shifted by ``action_scale``
                and ``action_bias``; ``new_carry`` is the memory carry only.
            """
            obs = jnp.expand_dims(obs, 0)
            carry = jnp.expand_dims(carry, 0)
            prev_action_batch = (
                jnp.expand_dims(prev_action, 0) if prev_action is not None else None
            )
            new_carry, x = nnx.merge(
                feature_extractor_graph, iqlearn.feature_extractor
            )(carry, obs, prev_action_batch)
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

        # Per-action Q/π diagnostic is discrete-only (all-actions critic).
        predict_qpi = None

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
        mask: jax.Array | None = None,
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
            mask: Optional action mask for ``latents`` (discrete only).

        Returns:
            ``(scalar_loss, metrics)`` where metrics contains ``"q"``,
            ``"entropy"`` and ``"v"``.
        """
        key_v = key
        mask_kw = {"mask": mask} if is_discrete else {}
        v, metrics = get_v(
            actor,
            critic,
            critic_graph,
            alpha,
            latents,
            key_v,
            include_entropy=True,
            include_log=True,
            **mask_kw,
        )
        metrics.update({"v": v.mean()})
        return -v.mean(), metrics

    def loss_actor_vtrace(
        actor: nnx.GraphState,
        critic: TwinCriticState,
        target_actor: nnx.GraphState,
        critic_target: TwinCriticState,
        actor_latents: jax.Array,
        target_latents: jax.Array,
        actions: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        behaviour_probs: jax.Array,
        alpha: jax.Array,
        key: jax.Array,
        masks: jax.Array | None = None,
    ) -> Tuple[jax.Array, dict]:
        """Off-policy V-trace (IMPALA-style) policy-gradient actor loss.

        Replaces the SAC ``max_π E_π[Q]`` objective (see :func:`loss_actor`).
        All inputs are time-major ``(T', B, …)`` — the same slices the critic
        and λ-critic losses consume.  The state-value baseline ``V(s) =
        Σ_a π(a|s) Q(s,a)`` and the V-trace return ``v_t`` are computed under
        the *target* actor + *target* critic (stable, fully stop-gradient'd);
        the only gradient path is through ``log π_online(a_t|s_t)`` on the
        online latents (so the actor still shapes the shared FE unless
        ``stop_actor_fe`` detached ``actor_latents`` upstream) and through the
        entropy bonus.  The trailing ``lambda_truncation`` steps are dropped
        (missing bootstrap mass at the unroll tail), matching the critics.

        Args:
            actor: Online actor head (the only differentiated network here).
            critic: Online twin-critic (unused; accepted for call-site
                symmetry with :func:`loss_actor`).
            target_actor: EMA actor for the baseline / V-trace value.
            critic_target: EMA twin-critic for the baseline / V-trace value.
            actor_latents: Online-FE features ``(T', B, feat)`` — already
                stop-gradient'd for the FE iff ``stop_actor_fe``.
            target_latents: Target-FE features ``(T', B, feat)``.
            actions: Time-major taken actions ``(T', B, 1)`` (float indices).
            rewards / dones / behaviour_probs: Time-major ``(T', B)``.
            alpha: Entropy temperature (reused as the PG entropy coefficient).
            key: PRNG key (unused on the discrete path).
            masks: Optional time-major action masks ``(T', B, action_dim)``.

        Returns:
            ``(scalar_loss, metrics)`` with ``"entropy"`` (so alpha autotune
            keeps working), ``"v"``, ``"q"`` and V-trace diagnostics.
        """
        assert is_discrete, "vtrace_actor is only supported for discrete actions."
        Tp, B = actor_latents.shape[:2]

        def _tb(a):
            return a.reshape((Tp * B,) + a.shape[2:])

        mask_flat = _tb(masks) if masks is not None else None

        # Baseline V(s_t) under the target policy/critic (hard value, no
        # entropy) — a scalar per state, far better conditioned than the
        # per-action Q differences the SAC actor chased.
        V = get_v(
            target_actor, critic_target, critic_graph, jnp.array(0.0),
            _tb(target_latents), key, include_entropy=False, mask=mask_flat,
        ).reshape(Tp, B)

        if params.fake_onpolicy_loss:
            ratios = jnp.ones_like(behaviour_probs)
        else:
            ratios = get_importance_ratios(
                target_actor, _tb(target_latents), _tb(actions),
                _tb(behaviour_probs), mask_flat,
            ).reshape(Tp, B)

        # V-trace return v_t (full trace, lam=1) — identical backward recursion
        # to loss_vtrace_lambda_sequence's scan_target.
        def scan_target(scan_carry, x):
            v_sp1, V_sp1 = scan_carry
            V_s, done, reward, ratio = x
            rho = jnp.minimum(params.rho_bar, ratio)
            c = jnp.minimum(params.c_bar, ratio)
            delta_V = reward + (1 - done) * params.gamma * V_sp1 - V_s
            v_s = V_s + rho * delta_V + (1 - done) * params.gamma * c * (v_sp1 - V_sp1)
            return (v_s, V_s), v_s

        _carry, vs = jax.lax.scan(
            scan_target,
            (jnp.zeros_like(V[0]), jnp.zeros_like(V[0])),
            (V, dones, rewards, ratios),
            reverse=True,
            unroll=8,
        )

        # Advantage Â_t = ρ_t (r_t + γ(1-d_t) v_{t+1} − V_t); v_{t+1} shifted
        # (the last entry is dropped by the lambda_truncation slice below).
        v_next = jnp.concatenate([vs[1:], vs[-1:]], axis=0)
        rho = jnp.minimum(params.rho_bar, ratios)
        adv = jax.lax.stop_gradient(
            rho * (rewards + (1 - dones) * params.gamma * v_next - V)
        )

        # Gradient path: log π_online(a_t|s_t) on the ONLINE latents.
        logits = _apply_mask(run_actor(actor, _tb(actor_latents)), mask_flat)
        logits = logits.reshape(Tp, B, -1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        probs = jax.nn.softmax(logits, axis=-1)
        idx = jnp.round(actions.reshape(Tp, B)).astype(jnp.int32)
        logp_a = jnp.take_along_axis(log_probs, idx[..., None], axis=-1)[..., 0]
        entropy = -(probs * log_probs).sum(-1)  # (T', B) exact H(π)

        L = params.lambda_truncation
        keep = slice(0, -L) if L > 0 else slice(None)
        adv_keep = adv[keep]
        adv_raw_mean = adv_keep.mean()  # keep the RAW mean as the offset diagnostic
        # --vtrace-center-advantage: subtract the batch-mean advantage (A2C/IMPALA
        # variance reduction). Removes the systematic offset (baseline lag) that
        # otherwise makes almost every advantage positive — restoring the
        # "suppress bad actions" half of the policy gradient so the policy can
        # actually sharpen instead of reinforcing whatever spread was sampled.
        if params.vtrace_normalize_advantage:
            # Full PPO-style standardisation (Â − mean)/std.  Under low-variance
            # returns the merely *centered* advantages are tiny in absolute
            # scale, so the raw PG gradient is negligible and the policy freezes
            # at ~uniform; dividing by the batch std restores a unit-scale
            # gradient.  Takes precedence over --vtrace-center-advantage.
            adv_keep = (adv_keep - adv_raw_mean) / (adv_keep.std() + 1e-8)
        elif params.vtrace_center_advantage:
            adv_keep = adv_keep - adv_raw_mean

        if params.ppo_clip_eps > 0.0:
            # PPO clipped surrogate = a trust region for the off-policy PG.
            # r_t = π_online(a_t|s_t) / μ(a_t|s_t) with μ the behaviour prob of
            # the taken action (the same quantity get_importance_ratios divides
            # by).  min() handles both advantage signs and caps the effective
            # step.  WITHOUT this the unclipped PG has no trust region, drives
            # one logit to +∞ and collapses the policy to a deterministic,
            # board-blind firing order within a few steps — which also destroys
            # the encoder's board memory (ladder: 0.865 with the clip vs 0.591
            # without, entropy 5e-4).
            log_ratio = logp_a - jnp.log(jnp.clip(behaviour_probs, 1e-8, None))
            ratio = jnp.exp(log_ratio)[keep]
            pg_loss = -jnp.minimum(
                ratio * adv_keep,
                jnp.clip(ratio, 1.0 - params.ppo_clip_eps, 1.0 + params.ppo_clip_eps)
                * adv_keep,
            ).mean()
            clip_frac = (jnp.abs(ratio - 1.0) > params.ppo_clip_eps).mean()
            ratio_mean = ratio.mean()
            # Per-timestep IS-ratio diagnostic (recurrent representational
            # drift): ratio[0] is at the window start (episode start, zero carry
            # → ≈1 unless a bookkeeping bug), ratio[-1] is deep into the episode
            # where the carry has been re-rolled many steps under updated
            # weights, so drift shows up there first.
            _lr_keep = log_ratio[keep]
            abslogr_t0 = jnp.abs(_lr_keep[0]).mean()
            abslogr_tlast = jnp.abs(_lr_keep[-1]).mean()
        else:
            pg_loss = -(adv_keep * logp_a[keep]).mean()
            clip_frac = jnp.array(0.0)
            ratio_mean = jnp.array(1.0)
            abslogr_t0 = abslogr_tlast = jnp.array(0.0)

        ent = entropy[keep].mean()
        loss = pg_loss - alpha * ent
        metrics = {
            "actor_pg_loss": pg_loss,
            "entropy": ent,
            "v": V[keep].mean(),
            "q": V[keep].mean(),
            "vtrace_adv": adv_raw_mean,
            "vtrace_v": vs[keep].mean(),
            "ppo_clip_frac": clip_frac,
            "ppo_ratio_mean": ratio_mean,
            "ppo_abslogratio_t0": abslogr_t0,
            "ppo_abslogratio_tlast": abslogr_tlast,
        }
        return loss, metrics

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
        next_mask: jax.Array | None = None,
        mask: jax.Array | None = None,
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
            next_mask: Optional action mask for the next state ``s'``
                (discrete only); used inside ``V(s')``.
            mask: Optional action mask for the CURRENT state ``s`` (discrete
                only); used only by the ``dense_value_coef`` term, which needs
                ``π(a|s)`` to weight the expectation over actions.

        Returns:
            ``(scalar_loss, metrics)`` where metrics contains
            ``"critic_loss"`` and ``"target_q"``.
        """
        key_v = key

        mask_kw = {"mask": next_mask} if is_discrete else {}
        next_v = get_v(
            actor_target,
            critic_target,
            critic_graph,
            alpha,
            target_latents,
            key_v,
            include_entropy=use_sac,
            **mask_kw,
        )
        target_q = jax.lax.stop_gradient(
            rewards + params.gamma * (1.0 - terminated) * next_v
        )

        q1, q2 = get_q_both(critic, critic_graph, latents, actions)
        loss = jnp.array(0.0)
        if params.sparse_value_loss:
            loss = loss + 0.5 * (jnp.mean((q1 - target_q) ** 2)
                                 + jnp.mean((q2 - target_q) ** 2))
        metrics = {"critic_loss": loss, "target_q": target_q.mean()}
        if params.dense_value_coef > 0.0 and is_discrete:
            # DENSE term (see dense_value_coef): the main SAC critic regresses
            # only Q(s,a_t), so like the λ-critics it feeds the shared FE through
            # one action column per step.  Regress the expected value
            # V(s) = Σ_a π(a|s)·min(Q1,Q2)(s,a) against the same 1-step target so
            # every column carries gradient.  π from the TARGET actor (no actor
            # perturbation); no entropy term, matching the λ-critic dense term.
            v_dense = get_v(
                actor_target,
                critic,
                critic_graph,
                jnp.array(0.0),
                latents,
                key_v,
                include_entropy=False,
                mask=mask,
            )
            l_dense = jnp.mean((v_dense - target_q) ** 2)
            loss = loss + params.dense_value_coef * l_dense
            metrics["critic_dense_loss"] = l_dense
            metrics["critic_loss"] = loss
        return loss, metrics

    def loss_ld(
        lambda1_critic: TwinCriticState,
        lambda2_critic: TwinCriticState,
        lambda1_graph: TwinCriticGraph,
        lambda2_graph: TwinCriticGraph,
        latents: jax.Array,
        actions: jax.Array,
        target_actor_state: nnx.GraphState | None = None,
        mask: jax.Array | None = None,
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

        # Freeze the λ-critic head params for this term so the discrepancy
        # gradient flows ONLY into the shared FE via ``latents`` — the
        # representation is what the λ-discrepancy is meant to regularise.
        # Without the stop_gradient the heads can trivially collapse q1≈q2 by
        # themselves (the cheapest descent direction), self-satisfying the
        # regulariser and leaving the FE — and thus ``lambda_coef`` — with no
        # effect.  The λ-critics still train via their V-trace losses.
        if params.dense_discrepancy:
            # DENSE discrepancy: compare the heads' EXPECTED values
            # V_i(s) = Σ_a π(a|s)·min(Q_i1,Q_i2)(s,a) instead of Q_i(s,a_t).
            # Same sparsity argument as dense_value_coef — at the executed
            # action only 1 of `action_dim` columns carries the discrepancy
            # gradient into the FE.  This is also what the reference does: its
            # λ-discrepancy term is on the scalar V heads, not per-action Q.
            assert target_actor_state is not None, (
                "dense_discrepancy needs the target actor to weight the "
                "expectation over actions"
            )
            v1 = get_v(target_actor_state, jax.lax.stop_gradient(lambda1_critic),
                       lambda1_graph, jnp.array(0.0), latents, jnp.array(0),
                       False, False, mask=mask)
            v2 = get_v(target_actor_state, jax.lax.stop_gradient(lambda2_critic),
                       lambda2_graph, jnp.array(0.0), latents, jnp.array(0),
                       False, False, mask=mask)
            loss = optax.losses.huber_loss(v1, v2).mean()
        else:
            q1 = get_q(jax.lax.stop_gradient(lambda1_critic), lambda1_graph, latents, actions)
            q2 = get_q(jax.lax.stop_gradient(lambda2_critic), lambda2_graph, latents, actions)
            loss = optax.losses.huber_loss(q1, q2).mean()
        return loss, {"ld_loss": loss}

    # ------------------------------------------------------------------
    # Online helpers: environment interaction and SAC update
    # ------------------------------------------------------------------

    def run_env_step(
        sac: SACState, env, env_params, env_state, env_carry, env_prev_action, key: jax.Array
    ):
        """Collect one transition from a gymnax environment into the online buffer.

        Threads the recurrent FE state across env steps: the memory carry and
        the prev-action encoding from the previous call are fed back into
        ``predict`` so the policy sees the full rollout history.  On episode
        termination both the carry and the prev-action are reset to zero
        (matching the per-episode reset performed during sequence training).

        Args:
            sac: Current agent state.  Only ``sac.online_buffer`` is mutated.
            env: Gymnax environment object (static — not traced by JAX).
            env_params: Gymnax environment parameters pytree.
            env_state: Current gymnax environment state pytree.
            env_carry: FE memory carry of shape ``(carry_dim,)`` from the
                previous step (zero on the first step of an episode).
            env_prev_action: Encoding of the action executed at the previous
                step, shape ``(action_dim,)`` (zero on the first step of an
                episode).  Width-0 / inert when ``use_prev_action=False``.
            key: JAX PRNG key; split internally for action sampling and env step.

        Returns:
            ``(new_sac, new_env_state, new_carry, new_prev_action)`` where
            ``new_carry`` / ``new_prev_action`` are the post-step FE state
            (both reset to zero on episode termination).
        """
        key_act, key_step, key_rb = jax.random.split(key, 3)
        obs = env.get_obs(env_state, env_params)
        pa = env_prev_action if use_prev_action else None
        if is_discrete:
            action, new_carry, prob = predict(
                sac, obs, env_carry, key_act, return_prob=True, prev_action=pa
            )
            if params.random_behaviour:
                # Override the executed action with a uniform draw over LEGAL
                # cells — a fixed random behaviour policy for isolating the
                # representation objective (e.g. GVD-only) on board-revealing,
                # order-invariant data with no learned-policy structure. The FE
                # carry from ``predict`` is kept (the memory still rolls over the
                # observation); only the env action / behaviour prob change.
                _mask = (mask_fn(obs) if mask_fn is not None
                         else jnp.ones((action_dim,), jnp.float32))
                _rb = jax.random.categorical(key_rb, jnp.where(_mask > 0, 0.0, -1e9))
                action = _rb.astype(jnp.float32)
                prob = 1.0 / jnp.maximum(_mask.sum(), 1.0)
            env_action = jnp.round(action).astype(jnp.int32)
        else:
            if approximate_lambda:
                action, new_carry, unsquashed_action = predict(
                    sac, obs, env_carry, key_act, return_unsquashed=True, prev_action=pa
                )
                # behaviour probability density under the squashed Gaussian
                _, logprob = sample_action_logprob(sac.actor, obs, key_act)[1:3]
                prob = jnp.exp(logprob)
            else:
                action, new_carry = predict(
                    sac, obs, env_carry, key_act, prev_action=pa
                )
            env_action = action
        # Encode the executed action for the next step's prev-action input.
        new_prev_action = (
            encode_action(jnp.atleast_1d(action))
            if use_prev_action
            else env_prev_action
        )
        _next_obs, new_env_state, reward, done, _ = env.step(
            key_step, env_state, env_action, env_params
        )
        transition = {
            obs_key: obs,
            action_key: jnp.atleast_1d(action),
            reward_key: jnp.asarray(reward, dtype=jnp.float32),
            terminated_key: jnp.asarray(done, dtype=jnp.float32),
        }
        if approximate_lambda or use_gvd:
            transition[behaviour_key] = prob
            if not is_discrete:
                transition[unsquashed_action_key] = jnp.atleast_1d(unsquashed_action)  # type: ignore
        if burn_in_from_stored_carry:
            # The PRE-step memory carry that ``predict`` consumed at this step.
            # After a terminal the threaded carry is zeroed below, so the
            # next slot stores zeros = episode start — correct for free.
            transition[carry_key] = env_carry
            if use_prev_action:
                # The PRE-step prev-action input (enc(a_{t-1})), zeroed by the
                # done reset below so the next slot encodes "episode start".
                transition[prev_action_key] = env_prev_action
        new_online_buffer = online_buffer_functions.add(
            sac.online_buffer, transition, terminated=done
        )
        new_carry = jax.lax.select(done, jnp.zeros_like(new_carry), new_carry)
        new_prev_action = jax.lax.select(
            done, jnp.zeros_like(new_prev_action), new_prev_action
        )
        return (
            sac._replace(online_buffer=new_online_buffer),
            new_env_state,
            new_carry,
            new_prev_action,
        )

    def calculate_latent(
        feature_extractor_state: nnx.GraphState,
        target_feature_extractor_state: nnx.GraphState,
        observations: jax.Array,
        actions: jax.Array,
        dones: jax.Array,
        init_carries: jax.Array,
        init_prev_actions: jax.Array | None = None,
    ):
        """Roll the online and target FEs over a sampled sequence.

        Performs an R2D2-style burn-in over the first ``params.burn_in_length``
        steps to warm the recurrent carry (gradients stopped at the burn-in
        boundary) and then unrolls the remainder to produce per-step latents.
        The carry is reset to zeros at any step where ``dones=True``.

        Online and target FEs share the same scan so they see the same carry
        reset pattern; both carries are initialised from ``init_carries`` and
        both consume the same per-step prev-action input.

        With ``use_prev_action`` the prev-action input is threaded inside the
        scan: step ``t`` consumes ``enc(a_{t-1})`` and emits ``enc(a_t)`` for
        ``t+1`` (zeroed by the done reset across episode boundaries) — matching
        the online path where ``run_env_step`` feeds the previously executed
        action back into ``predict``.  At the window start the prev-action is
        ``init_prev_actions`` (the stored value under stored-carry burn-in,
        else zero — the same approximation already made for the memory carry
        and absorbed by the burn-in).

        Args:
            feature_extractor_state: Online FE state (FE + memory).
            target_feature_extractor_state: Target FE state.
            observations: Batch-major observations of shape ``(B, T, *obs)``.
            actions: Batch-major stored actions of shape ``(B, T, ...)``.
                Only consumed when ``use_prev_action`` is enabled.
            dones: Per-step termination flags, shape ``(B, T)``.
            init_carries: Initial carries of shape ``(B, carry_dim)`` (zero
                for the start of each train() call).
            init_prev_actions: Initial prev-action input of shape
                ``(B, action_dim)`` for the window's first step.  ``None``
                defaults to zeros (no previous action).

        Returns:
            ``(latent, target_latent)``, both time-major with shape
            ``(T - burn_in_length, B, feature_dim)``.
        """
        feature_extractor = nnx.merge(feature_extractor_graph, feature_extractor_state)
        target_feature_extractor = nnx.merge(
            feature_extractor_graph, target_feature_extractor_state
        )
        _BL = params.burn_in_length

        # Strip the part of the stored observation the network must not see
        # (e.g. a packed action mask) before it reaches the FE.  ``obs_fn``
        # defaults to the identity.
        observations = obs_fn(observations)

        # Sample layout is (B, T, ...); lax.scan iterates leading axis, so move
        # the time axis to the front for the per-step inputs.
        observations = jnp.swapaxes(observations, 0, 1)
        dones = jnp.swapaxes(dones, 0, 1)

        if use_prev_action:
            # (T, B, action_dim) encodings of the action executed at each step.
            action_encs = encode_action(jnp.swapaxes(actions, 0, 1))
        else:
            # Zero-width dummy so the scan structure is identical either way.
            action_encs = jnp.zeros(
                (observations.shape[0], observations.shape[1], 0), dtype=jnp.float32
            )

        if init_prev_actions is None:
            init_prev_actions = jnp.zeros(
                (observations.shape[1], action_encs.shape[-1]), dtype=jnp.float32
            )

        carry_reset = jax.vmap(
            lambda done, carry: jax.lax.select(done, jnp.zeros_like(carry), carry)
        )

        def scan_carries(scan_carry, x):
            carry, target_carry, prev_action = scan_carry
            obs, done_step, action_enc = x
            done_step = done_step.astype(jnp.bool_)
            pa = prev_action if use_prev_action else None
            new_carry, y = feature_extractor(carry, obs, pa)
            new_target_carry, y_target = target_feature_extractor(
                target_carry, obs, pa
            )
            # Step t emits enc(a_t) as the prev-action input for step t+1; the
            # done reset zeroes it (and the carries) across episode boundaries.
            new_carry = carry_reset(done_step, new_carry)
            new_target_carry = carry_reset(done_step, new_target_carry)
            next_prev_action = carry_reset(done_step, action_enc)

            return (new_carry, new_target_carry, next_prev_action), (y, y_target)

        # Pair online + target carries; both start from the same zero tensor.
        burnt_in_carries, _ = jax.lax.scan(
            scan_carries,
            (init_carries, init_carries, init_prev_actions),
            (observations[:_BL], dones[:_BL], action_encs[:_BL]),
            unroll=8,  # fuse across timesteps -> fewer tiny per-step kernels
        )
        burnt_in_carries = jax.lax.stop_gradient(burnt_in_carries)

        _, latent = jax.lax.scan(
            scan_carries,
            burnt_in_carries,
            (observations[_BL:], dones[_BL:], action_encs[_BL:]),
            unroll=8,  # fuse across timesteps -> fewer tiny per-step kernels
        )

        return latent

    def _seq_loss_mean(per_step: jax.Array, dones: jax.Array) -> jax.Array:
        """Reduce a per-step sequence loss ``(T', B)`` to a scalar.

        Drops the trailing ``params.lambda_truncation`` steps (their λ-return is
        biased by the missing bootstrap mass at the unroll tail) and, when
        ``params.mask_first_episode_only`` is set, every step after the window's
        first episode.

        The mask matters with ``episode_aligned_sampling``: a window starts at an
        episode start, so a short episode's window spills into the *next*
        episode's opening steps, over-representing early/low-memory states and
        biasing the FE gradient away from the memory-rich late ones.

        ``dones[t]`` is True iff step *t* terminated, so
        ``cumsum(dones) - dones`` counts terminations strictly *before* t and the
        first episode is ``== 0`` (i.e. the terminal step itself is kept).

        Args:
            per_step: Per-step loss, shape ``(T', B)``.
            dones: Per-step termination flags, shape ``(T', B)``.

        Returns:
            Scalar mean over the kept steps.
        """
        L = params.lambda_truncation
        keep = slice(0, -L) if L > 0 else slice(None)
        x = per_step[keep]
        if not params.mask_first_episode_only:
            return x.mean()
        prior_dones = jnp.cumsum(dones, axis=0) - dones
        m = (prior_dones == 0).astype(x.dtype)[keep]
        return jnp.sum(x * m) / jnp.clip(jnp.sum(m), 1.0)

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
        masks: jax.Array | None = None,
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
            key: JAX PRNG key (unused on the discrete path; kept for API
                compatibility).
            masks: Optional time-major action masks, shape
                ``(T', B, action_dim)``; aligned with ``latents``.  ``None``
                disables masking.

        Returns:
            ``(scalar_loss, metrics)``.
        """
        assert is_discrete, (
            "loss_vtrace_lambda_sequence does not yet support continuous action "
            "spaces: importance ratios would require unsquashed (pre-tanh) actions, "
            "which are not currently wired through."
        )

        # Forward pass has no time recurrence — evaluate all T' steps as one
        # big batch (one matmul per head instead of T' sequential scan
        # iterations; the GPU-relevant de-scan).  ``get_q`` assumes a single
        # leading batch dim, so flatten (T', B) -> (T'·B) and reshape back.
        Tp, B = latents.shape[:2]

        def _tb(a):
            return a.reshape((Tp * B,) + a.shape[2:])

        mask_flat = _tb(masks) if masks is not None else None
        v = get_v(
            target_actor_state,
            q_target_state,
            q_graph,
            jnp.array(0.0),
            _tb(target_latents),
            key,
            False,
            False,
            mask=mask_flat,
        ).reshape(Tp, B)
        q = get_q(q_state, q_graph, _tb(latents), _tb(actions)).reshape(Tp, B)
        if params.fake_onpolicy_loss:
            ratios = jnp.ones_like(behaviour_probs)
        else:
            ratios = get_importance_ratios(
                target_actor_state,
                _tb(target_latents),
                _tb(actions),
                _tb(behaviour_probs),
                mask_flat,
            ).reshape(Tp, B)

        # Backward V-trace recursion — inherently sequential (recursive in
        # time) but elementwise-cheap; stays a scan.
        def scan_target(scan_carry, x):
            v_sp1, V_sp1 = scan_carry
            V_s, done, reward, ratio = x
            rho = jnp.minimum(params.rho_bar, ratio)
            c = lam * jnp.minimum(params.c_bar, ratio)
            delta_V = reward + (1 - done) * params.gamma * V_sp1 - V_s
            v_s = V_s + rho * delta_V + (1 - done) * params.gamma * c * (v_sp1 - V_sp1)
            return (v_s, V_s), v_s

        _carry, targets = jax.lax.scan(
            scan_target,
            (jnp.zeros_like(v[0]), jnp.zeros_like(v[0])),
            (v, dones, rewards, ratios),
            reverse=True,
            unroll=8,  # fuse across timesteps -> fewer tiny per-step kernels
        )

        tgt = jax.lax.stop_gradient(targets)
        metrics = {
            f"lambda{lam}_critic:": q[: -params.lambda_truncation].mean(),
            f"lambda{lam}_target:": targets[: -params.lambda_truncation].mean(),
        }
        loss = jnp.array(0.0)
        if params.sparse_value_loss:
            loss = loss + _seq_loss_mean(optax.losses.huber_loss(q, tgt), dones)
        if params.dense_value_coef > 0.0:
            # DENSE term: regress the ONLINE head's expected value
            # V(s) = Σ_a π(a|s)·min(Q1,Q2)(s,a) against the same V-trace target,
            # so every action column (and hence the shared FE) gets gradient
            # every step instead of only the executed action's column.  π comes
            # from the TARGET actor, so this stays a pure critic/FE term.
            # min-over-twins matches how the target's V(s) is built above.
            v_dense = get_v(
                target_actor_state,
                q_state,
                q_graph,
                jnp.array(0.0),
                _tb(latents),
                key,
                False,
                False,
                mask=mask_flat,
            ).reshape(Tp, B)
            l_dense = _seq_loss_mean(optax.losses.huber_loss(v_dense, tgt), dones)
            loss = loss + params.dense_value_coef * l_dense
            metrics[f"lambda{lam}_dense_loss"] = l_dense
            metrics[f"lambda{lam}_dense_v"] = v_dense[: -params.lambda_truncation].mean()
        metrics["loss"] = loss

        return loss, metrics

    # ------------------------------------------------------------------
    # GVD: successor-feature V-trace + discrepancy (Koepernik et al. 2025)
    # ------------------------------------------------------------------

    if use_gvd:
        def run_sf(sf_state: nnx.GraphState, sf_graph, x: jax.Array) -> jax.Array:
            """Run one SF head: per-action successor features.

            The head's flat output is reshaped to one feature vector per
            action (all-actions layout, like the discrete λ-critics).

            Args:
                sf_state: SF head parameter state.
                sf_graph: Matching graph def (``gvd_sf{1,2}_graph``).
                x: Features ``(batch, feature_dim)``.

            Returns:
                SF values ``(batch, action_dim, n_gvd_features)``.
            """
            out = nnx.merge(sf_graph, sf_state)(x)
            return out.reshape(*out.shape[:-1], action_dim, n_gvd_features)

        def run_sf_q(sf_state: nnx.GraphState, sf_graph, x: jax.Array, actions: jax.Array) -> jax.Array:
            """SF analogue of Q(s, a): gather the executed action's row.

            Returns ``(batch, n_gvd_features)``.
            """
            sf_q = run_sf(sf_state, sf_graph, x)  # (batch, A, n)
            indices = jnp.round(actions.reshape(-1)).astype(jnp.int32)
            batch = sf_q.shape[0]
            return sf_q[jnp.arange(batch), indices]

        def run_sf_v(actor, sf_state: nnx.GraphState, sf_graph, x: jax.Array, mask: jax.Array) -> jax.Array:
            """SF analogue of V(s): policy expectation over per-action SFs.

            No minimum over twins (features have no max objective attached),
            no entropy term — a plain expectation under the (masked) policy.

            Returns ``(batch, n_gvd_features)``.
            """
            logits = _apply_mask(run_actor(actor, x), mask)  # (batch, A)
            probs = jax.nn.softmax(logits)                   # (batch, A)
            sf_q = run_sf(sf_state, sf_graph, x)             # (batch, A, n)
            return jnp.sum(probs[..., None] * sf_q, axis=-2)  # (batch, n)

        def loss_vtrace_sf_sequence(
            sf_state: nnx.GraphState,
            sf_target_state: nnx.GraphState,
            sf_graph,
            target_actor_state: nnx.GraphState,
            lam: float,
            tag: str,
            actions: jax.Array,
            cumulants: jax.Array,
            dones: jax.Array,
            behaviour_probs: jax.Array,
            latents: jax.Array,
            target_latents: jax.Array,
            masks: jax.Array | None = None,
        ) -> Tuple[jax.Array, dict]:
            """V-trace TD loss for one SF head over a time-major sequence.

            Q-style port of ``loss_vtrace_lambda_sequence`` to vector
            (per-feature) values: per step, ``V(s)`` is the policy
            expectation over the TARGET head's per-action SFs (``run_sf_v``,
            no twin minimum — features have no max objective attached) and
            the online head's SF at the executed action (``run_sf_q``) is
            regressed onto the V-trace λ-return targets built from the
            ``cumulants`` (which occupy the reward slot of the recursion;
            see :func:`sf_vtrace_targets`).  Huber per feature, summed over
            features, mean elsewhere; last ``params.lambda_truncation``
            steps dropped.

            Metrics: ``gvd_{tag}_loss``, ``gvd_{tag}_critic:``,
            ``gvd_{tag}_target:`` (colon convention matching the scalar
            λ-critic metrics).

            Returns:
                ``(scalar_loss, metrics_dict)``.
            """
            assert is_discrete, (
                "loss_vtrace_sf_sequence does not yet support continuous action "
                "spaces: importance ratios would require unsquashed (pre-tanh) actions, "
                "which are not currently wired through."
            )

            # Forward pass has no time recurrence — evaluate all T' steps as
            # one big batch (de-scan; see loss_vtrace_lambda_sequence).
            # ``run_sf_q`` assumes a single leading batch dim, so flatten
            # (T', B) -> (T'·B) and reshape back.
            Tp, B = latents.shape[:2]

            def _tb(a):
                return a.reshape((Tp * B,) + a.shape[2:])

            mask_flat = _tb(masks) if masks is not None else None
            v = run_sf_v(
                target_actor_state,
                sf_target_state,
                sf_graph,
                _tb(target_latents),
                mask_flat,
            ).reshape(Tp, B, -1)
            q = run_sf_q(
                sf_state, sf_graph, _tb(latents), _tb(actions)
            ).reshape(Tp, B, -1)
            if params.fake_onpolicy_loss:
                ratios = jnp.ones_like(behaviour_probs)
            else:
                ratios = get_importance_ratios(
                    target_actor_state,
                    _tb(target_latents),
                    _tb(actions),
                    _tb(behaviour_probs),
                    mask_flat,
                ).reshape(Tp, B)

            # Backward V-trace recursion over (T', B, n) — shared,
            # module-level kernel (unit-tested against a hand-rolled
            # reference in tests/test_gvd.py).
            targets = sf_vtrace_targets(
                v,
                cumulants,
                dones,
                ratios,
                params.gamma,
                lam,
                params.rho_bar,
                params.c_bar,
            )

            tgt = jax.lax.stop_gradient(targets)
            metrics = {
                f"gvd_{tag}_critic:": q[: -params.lambda_truncation].mean(),
                f"gvd_{tag}_target:": targets[: -params.lambda_truncation].mean(),
            }
            # sum over the SF feature axis, then mean over the kept steps
            loss = jnp.array(0.0)
            if params.sparse_value_loss:
                loss = loss + _seq_loss_mean(
                    optax.losses.huber_loss(q, tgt).sum(-1), dones)
            if params.dense_value_coef > 0.0:
                # DENSE term (SF analogue of the λ-critic one): regress the
                # ONLINE head's expected SF Σ_a π(a|s)·SF(s,a) against the same
                # V-trace target, so every action column feeds the shared FE
                # instead of only the executed action's.
                v_dense = run_sf_v(
                    target_actor_state,
                    sf_state,
                    sf_graph,
                    _tb(latents),
                    mask_flat,
                ).reshape(Tp, B, -1)
                l_dense = _seq_loss_mean(
                    optax.losses.huber_loss(v_dense, tgt).sum(-1), dones)
                loss = loss + params.dense_value_coef * l_dense
                metrics[f"gvd_{tag}_dense_loss"] = l_dense
            metrics[f"gvd_{tag}_loss"] = loss

            return loss, metrics

        def loss_gvd(
            gvd_sf1_state: nnx.GraphState,
            gvd_sf2_state: nnx.GraphState,
            latents_flat: jax.Array,
            actions_flat: jax.Array,
            target_actor_state: nnx.GraphState | None = None,
            mask: jax.Array | None = None,
        ) -> Tuple[jax.Array, dict]:
            """GVD discrepancy between the two SF heads at the stored actions.

            Both head parameter states are frozen via
            ``jax.lax.stop_gradient`` (mirror of ``loss_ld``): the
            discrepancy gradient reaches only the shared FE via
            ``latents_flat``.  Huber between the per-action SFs of the two
            λ heads — the SF analogue of the Q-based λ-discrepancy.

            Args:
                gvd_sf1_state: Online SF1 head state.
                gvd_sf2_state: Online SF2 head state.
                latents_flat: ``(T'·B, feature_dim)`` flattened online-FE
                    features (caller passes ``_flat(latent[:-1])``).
                actions_flat: ``(T'·B, 1)`` executed actions at those states.

            Returns:
                ``(scalar_loss, {"gvd_loss": loss})``.
            """
            if params.dense_discrepancy:
                # DENSE SF discrepancy — see loss_ld / dense_value_coef: at the
                # executed action only 1 of `action_dim` columns carries the
                # discrepancy gradient into the shared FE.
                assert target_actor_state is not None, (
                    "dense_discrepancy needs the target actor to weight the "
                    "expectation over actions"
                )
                sf_q1 = run_sf_v(target_actor_state,
                                 jax.lax.stop_gradient(gvd_sf1_state),
                                 gvd_sf1_graph, latents_flat, mask)
                sf_q2 = run_sf_v(target_actor_state,
                                 jax.lax.stop_gradient(gvd_sf2_state),
                                 gvd_sf2_graph, latents_flat, mask)
            else:
                sf_q1 = run_sf_q(jax.lax.stop_gradient(gvd_sf1_state), gvd_sf1_graph, latents_flat, actions_flat)
                sf_q2 = run_sf_q(jax.lax.stop_gradient(gvd_sf2_state), gvd_sf2_graph, latents_flat, actions_flat)
            loss = optax.losses.huber_loss(sf_q1, sf_q2).mean()
            # NB: "gvd_loss", not "ld_loss" — the λ-discrepancy already emits
            # ld_loss and metrics.update would silently collide.
            return loss, {"gvd_loss": loss}

    def loss_combined(
        feature_extractor_state,
        actor_state,
        critic_state,
        lambda1_critic_state,
        lambda2_critic_state,
        gvd_sf1_state,
        gvd_sf2_state,
        target_feature_extractor_state,
        actor_target_state,
        critic_target_state,
        lambda1_critic_target_state,
        lambda2_critic_target_state,
        gvd_sf1_target_state,
        gvd_sf2_target_state,
        alpha,
        buffer: Buffer,
        key: jax.Array,
    ):
        """Joint R2D2-style sequence loss: actor + critic + (optional) λ-critics + GVD.

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
        - **GVD SF V-trace losses + discrepancy** (when ``use_gvd=True``):
          two successor-feature heads trained on feature-difference
          cumulants at ``params.gvd_lambda{1,2}``, plus the squared SF
          discrepancy ``loss_gvd`` scaled by ``params.gvd_coef``.

        ``jax.grad(loss_combined, argnums=[0,1,2,3,4,5,6])`` distributes
        gradients to the FE / actor / critic / λ1-critic / λ2-critic /
        SF1 / SF2 respectively; target nets are passed positionally outside
        ``argnums`` so JAX treats them as constants (no leakage).

        Returns:
            ``(scalar_loss, metrics_dict)``.
        """
        _BL = params.burn_in_length

        key_sample, key_actor, key_critic, key_lambda_critic1, key_lambda_critic2 = (
            jax.random.split(key, 5)
        )
        sample, indices = online_buffer_lambda_sample(buffer, key_sample)
        if burn_in_from_stored_carry:
            # R2D2 "stored state": start the burn-in from the carry the
            # online policy actually consumed at the window's first step.
            # Stale w.r.t. the current FE weights — the burn-in refreshes
            # it.  Pre-fill / never-written slots hold zeros (episode-start
            # semantics), matching the zero-init fallback below.
            init_carries = sample.this_info[carry_key][:, 0]
            # Prev-action consumed at the window's first step (zeros for
            # never-written / episode-start slots); None when prev-action is
            # disabled so calculate_latent falls back to its zero default.
            init_prev_actions = (
                sample.this_info[prev_action_key][:, 0] if use_prev_action else None
            )
        else:
            init_carries = feature_extractor.initialize_carry(
                params.online_batch_size
            )
            init_prev_actions = None

        observations = sample.this_info[obs_key]
        actions = sample.this_info[action_key]
        rewards = sample.this_info[reward_key]
        terminated = sample.this_info[terminated_key]
        behaviour = sample.this_info[behaviour_key]

        # Action masks are derived from the *full* stored observation (the FE
        # never sees them — see ``calculate_latent``).  ``None`` when masking
        # is disabled.
        masks = mask_fn(observations) if (is_discrete and mask_fn is not None) else None

        latent, target_latent = calculate_latent(
            feature_extractor_state,
            target_feature_extractor_state,
            observations,
            actions,
            terminated,
            init_carries,
            init_prev_actions,
        )
        # latent / target_latent shapes: (T - _BL, B, feat) -- time-major.

        # Sample fields are (B, T, ...); convert to time-major and slice down to
        # the BPTT (post-burn-in) window so they align with the latents.
        actions_tm = jnp.swapaxes(actions, 0, 1)[_BL:]
        rewards_tm = jnp.swapaxes(rewards, 0, 1)[_BL:]
        terminated_tm = jnp.swapaxes(terminated, 0, 1)[_BL:]
        behaviour_tm = jnp.swapaxes(behaviour, 0, 1)[_BL:]
        masks_tm = jnp.swapaxes(masks, 0, 1)[_BL:] if masks is not None else None

        def _flat(x):
            # (T', B, ...) -> (T' * B, ...)
            return x.reshape((-1, *x.shape[2:]))

        # Actor: per-state objective, treat (t, b) as an independent batch.
        # Use the *online* critic but freeze its params via stop_gradient so
        # the actor loss does not train the critic. By default the gradient also
        # flows into the shared FE via the latent input. --stop-actor-fe detaches
        # that path (SAC-AE recipe): the FE is then trained only by the value /
        # critic (and GVD) losses, and the actor is a head on frozen features.
        # This removes the actor's E_π[Q]-maximising pressure on the shared
        # recurrent memory (a suspected driver of value-magnitude inflation)
        # while preserving GVD's influence on the representation.
        actor_latent = (jax.lax.stop_gradient(latent)
                        if params.stop_actor_fe else latent)
        if params.vtrace_actor:
            # Off-policy V-trace policy gradient (IMPALA-style) instead of the
            # SAC E_π[Q] objective. Consumes the time-major sequence directly.
            l_actor, metrics = loss_actor_vtrace(
                actor_state,
                jax.lax.stop_gradient(critic_state),
                actor_target_state,
                critic_target_state,
                actor_latent,
                target_latent,
                actions_tm,
                rewards_tm,
                terminated_tm,
                behaviour_tm,
                alpha,
                key_actor,
                masks=masks_tm,
            )
        else:
            l_actor, metrics = loss_actor(
                actor_state,
                jax.lax.stop_gradient(critic_state),
                _flat(actor_latent),
                alpha,
                key_actor,
                mask=_flat(masks_tm) if masks_tm is not None else None,
            )

        # --stop-critic-fe: detach the online FE latents feeding the value heads
        # (mirror of gvd_stop_fe / stop_actor_fe). The critic still learns; its
        # loss no longer backprops into the shared FE via `latent`. target_latent
        # is already the target FE and is left untouched.
        critic_latent = (jax.lax.stop_gradient(latent)
                         if params.stop_critic_fe else latent)

        # Critic: pair (latent[t], target_latent[t+1]) so V(s') is computed at
        # the next state. Action / reward / terminated at index t.
        l_critic, metrics_critic = loss_critic(
            actor_target_state,
            critic_state,
            critic_target_state,
            _flat(critic_latent[:-1]),
            _flat(target_latent[1:]),
            _flat(actions_tm[:-1]),
            rewards_tm[:-1].reshape(-1),
            terminated_tm[:-1].reshape(-1),
            alpha,
            key_critic,
            next_mask=_flat(masks_tm[1:]) if masks_tm is not None else None,
            mask=_flat(masks_tm[:-1]) if masks_tm is not None else None,
        )
        metrics.update(metrics_critic)

        loss = l_actor + params.sac_critic_coef * l_critic
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
                critic_latent,
                target_latent,
                key_lambda_critic1,
                masks=masks_tm,
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
                critic_latent,
                target_latent,
                key_lambda_critic2,
                masks=masks_tm,
            )
            metrics.update(metrics_lambda2_critic)

            l_ld, metrics_ld = loss_ld(
                lambda1_critic_state,
                lambda2_critic_state,
                lambda1_critic_graph,
                lambda2_critic_graph,
                _flat(critic_latent[:-1]),
                _flat(actions_tm[:-1]),
                actor_target_state,
                _flat(masks_tm[:-1]) if masks_tm is not None else None,
            )
            metrics.update(metrics_ld)
            loss += l_lambda1 + l_lambda2 + params.lambda_coef* l_ld

        if use_gvd:
            # φ over the RAW stored obs (B, T, obs) — fixed map, no params.
            # The previous action a_{t-1} is paired with o_t so the cumulant
            # can localise the per-step hit bit (= result of a_{t-1}) to the
            # cell that was fired.  a_{t-1} is obtained by shifting the action
            # stream one step along time; index 0 has no predecessor so it is
            # zeroed (it falls in the dropped burn-in window, and at any
            # episode boundary o_t[hit_bit]=0 gates the stale action to nothing).
            prev_actions = jnp.concatenate(
                [jnp.zeros_like(actions[:, :1]), actions[:, :-1]], axis=1
            )
            phi = gvd_feature_fn(observations, prev_actions)  # (B, T, n)
            phi_tm = jnp.swapaxes(phi, 0, 1)[_BL:]           # (T', B, n)
            # Cumulant f_t = φ(o_t, a_{t-1}): the paper's pseudo-return is
            # Σ γ^t f(ω_t) directly (Eq. 4/6), so the features go into the
            # reward slot of the V-trace recursion.  --gvd-cumulant-diff swaps
            # in the Jaderberg-style temporal difference f_t = φ(o_{t+1})−φ(o_t)
            # (the paper's SF-collapse remedy), zeroed across episode boundaries
            # so a terminal→reset transition contributes no spurious jump.
            if params.gvd_cumulant_diff:
                phi_next = jnp.concatenate([phi_tm[1:], phi_tm[-1:]], axis=0)
                cumulants = (phi_next - phi_tm) * (1.0 - terminated_tm)[..., None]
            else:
                cumulants = phi_tm
            # Bound the cumulant magnitude to keep the SF TD targets small and
            # stop sf-head divergence (scale≈(1-γ) → SF in [0,1]). Default 1.0.
            cumulants = cumulants * params.gvd_cumulant_scale

            # --gvd-stop-fe: sever GVD/SF gradients from the shared FE. The SF
            # heads still train (their targets come from target_latent, which is
            # already stop-gradded), but their loss no longer backprops into the
            # online FE via `latent`. Strictly stronger than gvd_coef=0, which
            # leaves the SF heads' own TD loss training the FE.
            sf_latent = (jax.lax.stop_gradient(latent)
                         if params.gvd_stop_fe else latent)

            l_sf1, m_sf1 = loss_vtrace_sf_sequence(
                gvd_sf1_state,
                gvd_sf1_target_state,
                gvd_sf1_graph,
                actor_target_state,
                params.gvd_lambda1,
                "sf1",
                actions_tm,
                cumulants,
                terminated_tm,
                behaviour_tm,
                sf_latent,
                target_latent,
                masks=masks_tm,
            )
            metrics.update(m_sf1)
            l_sf2, m_sf2 = loss_vtrace_sf_sequence(
                gvd_sf2_state,
                gvd_sf2_target_state,
                gvd_sf2_graph,
                actor_target_state,
                params.gvd_lambda2,
                "sf2",
                actions_tm,
                cumulants,
                terminated_tm,
                behaviour_tm,
                sf_latent,
                target_latent,
                masks=masks_tm,
            )
            metrics.update(m_sf2)

            l_gvd, m_gvd = loss_gvd(
                gvd_sf1_state,
                gvd_sf2_state,
                _flat(sf_latent[:-1]),
                _flat(actions_tm[:-1]),
                actor_target_state,
                _flat(masks_tm[:-1]) if masks_tm is not None else None,
            )
            metrics.update(m_gvd)
            loss += l_sf1 + l_sf2 + params.gvd_coef * l_gvd

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
            per-λ tags, (when ``use_gvd``) ``"gvd_loss"`` and the
            ``gvd_sf{1,2}_*`` tags, and ``"alpha"`` (when
            ``params.autotune_alpha``).
        """

        (
            grads_fe,
            grads_actor,
            grads_critic,
            grads_lambda1_critic,
            grads_lambda2_critic,
            grads_gvd_sf1,
            grads_gvd_sf2,
        ), metrics = jax.grad(
            loss_combined, argnums=[0, 1, 2, 3, 4, 5, 6], has_aux=True
        )(
            sac.feature_extractor,
            sac.actor,
            sac.critic,
            sac.lambda1_critic,
            sac.lambda2_critic,
            sac.gvd_sf1,
            sac.gvd_sf2,
            sac.feature_extractor_target,
            sac.actor_target,
            sac.critic_target,
            sac.lambda1_critic_target,
            sac.lambda2_critic_target,
            sac.gvd_sf1_target,
            sac.gvd_sf2_target,
            sac.alpha,
            sac.online_buffer,
            key,
        )

        updates, new_fe_opt = fe_optimizer.update(grads_fe, sac.fe_optimizer_state)
        new_fe = optax.apply_updates(sac.feature_extractor, updates)  # type: ignore

        updates, new_actor_opt = actor_optimizer.update(
            grads_actor, sac.actor_optimizer_state
        )
        new_actor = optax.apply_updates(sac.actor, updates)  # type: ignore

        updates, new_critic_opt = critic_optimizer.update(
            grads_critic, sac.critic_optimizer_state
        )
        new_critic = optax.apply_updates(sac.critic, updates)  # type: ignore

        if approximate_lambda:
            updates, new_lambda1_critic_opt = lambda1_critic_optimizer.update(
                grads_lambda1_critic, sac.lambda1_critic_optimizer_state
            )
            new_lambda1_critic = optax.apply_updates(sac.lambda1_critic, updates)  # type: ignore
            updates, new_lambda2_critic_opt = lambda2_critic_optimizer.update(
                grads_lambda2_critic, sac.lambda2_critic_optimizer_state
            )
            new_lambda2_critic = optax.apply_updates(sac.lambda2_critic, updates)  # type: ignore

        if use_gvd:
            updates, new_gvd_sf1_opt = gvd_sf1_optimizer.update(
                grads_gvd_sf1, sac.gvd_sf1_optimizer_state
            )
            new_gvd_sf1 = optax.apply_updates(sac.gvd_sf1, updates)  # type: ignore
            updates, new_gvd_sf2_opt = gvd_sf2_optimizer.update(
                grads_gvd_sf2, sac.gvd_sf2_optimizer_state
            )
            new_gvd_sf2 = optax.apply_updates(sac.gvd_sf2, updates)  # type: ignore

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

        update_param = jax.lax.select((sac.update_step + 1) % params.periodic_update == 0, params.tau, 0.0)
        new_fe_target = jax.tree.map(
            lambda x, y: (1 - update_param) * x + update_param * y,
            sac.feature_extractor_target,
            new_fe,
        )
        new_actor_target = jax.tree.map(
            lambda x, y: (1 - update_param) * x + update_param * y,
            sac.actor_target,
            new_actor,
        )
        new_critic_target = jax.tree.map(
            lambda x, y: (1 - update_param) * x + update_param * y,
            sac.critic_target,
            new_critic,
        )
        if approximate_lambda:
            new_lambda1_critic_target = jax.tree.map(
                lambda x, y: (1 - update_param) * x + update_param * y,
                sac.lambda1_critic_target,
                new_lambda1_critic,
            )
            new_lambda2_critic_target = jax.tree.map(
                lambda x, y: (1 - update_param) * x + update_param * y,
                sac.lambda2_critic_target,
                new_lambda2_critic,
            )
        if use_gvd:
            new_gvd_sf1_target = jax.tree.map(
                lambda x, y: (1 - update_param) * x + update_param * y,
                sac.gvd_sf1_target,
                new_gvd_sf1,
            )
            new_gvd_sf2_target = jax.tree.map(
                lambda x, y: (1 - update_param) * x + update_param * y,
                sac.gvd_sf2_target,
                new_gvd_sf2,
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
                sac.update_step + 1,
                gvd_sf1=new_gvd_sf1 if use_gvd else sac.gvd_sf1,
                gvd_sf2=new_gvd_sf2 if use_gvd else sac.gvd_sf2,
                gvd_sf1_target=(
                    new_gvd_sf1_target if use_gvd else sac.gvd_sf1_target
                ),
                gvd_sf2_target=(
                    new_gvd_sf2_target if use_gvd else sac.gvd_sf2_target
                ),
                gvd_sf1_optimizer_state=(
                    new_gvd_sf1_opt if use_gvd else sac.gvd_sf1_optimizer_state
                ),
                gvd_sf2_optimizer_state=(
                    new_gvd_sf2_opt if use_gvd else sac.gvd_sf2_optimizer_state
                ),
            ),
            metrics,
        )

    # Single-row (unbatched) zero carry for the env-rollout policy queries.
    _env_zero_carry = feature_extractor.initialize_carry(1)[0]
    # Zero prev-action input (width-0 / inert when use_prev_action=False).
    _env_zero_prev_action = feature_extractor.initialize_prev_action(1)[0]

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

        The prev-action input (when ``use_prev_action`` is enabled) is threaded
        internally and started from zero at each call, mirroring how the memory
        carry is reset fresh per :func:`train` invocation; within the scan it is
        threaded step-to-step and reset on episode boundaries.
        """

        def scan_fun(carry, _):
            sac, env_state, env_carry, env_prev_action, key = carry
            key, next_key, env_key, update_key = jax.random.split(key, 4)
            sac, env_state, env_carry, env_prev_action = run_env_step(
                sac, env, env_params, env_state, env_carry, env_prev_action, env_key
            )
            sac, metrics = update_step(sac, update_key)
            return (sac, env_state, env_carry, env_prev_action, next_key), metrics

        (sac, env_state, env_carry, _, _), metrics = jax.lax.scan(
            scan_fun,
            (sac, env_state, env_carry, _env_zero_prev_action, key),
            length=train_steps,
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
                if mask_fn is not None:
                    # Uniform over *legal* actions only, so the behaviour policy
                    # (and its stored probability) is consistent with the masked
                    # target policy used in the V-trace importance ratios.
                    mask = mask_fn(obs).astype(bool)
                    legal_logits = jnp.where(mask, 0.0, -jnp.inf)
                    action_idx = jax.random.categorical(key_act, legal_logits)
                    prob = 1.0 / jnp.maximum(jnp.sum(mask), 1).astype(jnp.float32)
                else:
                    action_idx = jax.random.randint(key_act, (), 0, action_dim)
                    prob = jnp.float32(1.0 / action_dim)
                action = action_idx.astype(jnp.float32)
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
            if approximate_lambda or use_gvd:
                transition[behaviour_key] = prob
                if not is_discrete:
                    transition[unsquashed_action_key] = jnp.atleast_1d(u)
            # With burn_in_from_stored_carry the carry slot is deliberately
            # NOT written here: the random pre-fill policy threads no
            # memory, and the buffer-init zeros mean "episode start".
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
        predict,
        train,
        get_importance_ratios,
        prefill_buffer,
        _train_unrolled,
        encode_action,
    )
    if debug:
        _debug_graphs = {"critic": critic_graph}
        if approximate_lambda:
            _debug_graphs["lambda1"] = lambda1_critic_graph
            _debug_graphs["lambda2"] = lambda2_critic_graph
        if use_gvd:
            _debug_graphs["gvd_sf1"] = gvd_sf1_graph
            _debug_graphs["gvd_sf2"] = gvd_sf2_graph
        return (
            iqlearn,
            fns,
            DebugFunctions(get_q, get_entropy, calculate_latent, predict_qpi,
                           get_v, _debug_graphs),
        )
    return (iqlearn, fns)
