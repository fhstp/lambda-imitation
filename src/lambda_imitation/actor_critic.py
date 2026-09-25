"""Discrete recurrent actor–critic with auxiliary Retrace discrepancy.

The task critic fits reward-only one-step targets with MSE. Two optional twin
critics fit Retrace targets with Huber regression (threshold 1). Their squared
disagreement trains the shared memory, with the auxiliary heads held fixed.
The entropy-regularised actor reads a detached task critic. Its gradients stop
at recurrent memory; the memoryless control still trains its input embedding.
All training state is immutable; factories return pure, JAX-compatible closures.
"""

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from .buffer import Buffer, create_buffer, create_sequence_sample

_nnx_data = getattr(nnx, "data", lambda x: x)
reward_key = "rewards"
terminated_key = "terminated"
behaviour_key = "behaviour_weight"


class Head(nnx.Module):
    """MLP with ReLU hidden layers and a linear all-actions output."""

    def __init__(self, feature_dim, hidden_dims, output_dim, *,
                 layer_norm=False, rngs):
        dims = [feature_dim, *hidden_dims, output_dim]
        self.layers = _nnx_data([
            nnx.Linear(dims[i], dims[i + 1], rngs=rngs)
            for i in range(len(dims) - 1)
        ])
        self.norms = _nnx_data([
            nnx.LayerNorm(d, rngs=rngs) for d in hidden_dims
        ] if layer_norm else [])

    def __call__(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.norms:
                x = self.norms[i](x)
            x = nnx.relu(x)
        return self.layers[-1](x)


class TwinCriticState(NamedTuple):
    q1: nnx.GraphState
    q2: nnx.GraphState


class TwinCriticGraph(NamedTuple):
    q1: nnx.GraphDef
    q2: nnx.GraphDef


class AgentState(NamedTuple):
    """Networks, EMA targets, Adam states and replay for one training seed."""

    feature_extractor: nnx.GraphState
    actor: nnx.GraphState
    critic: TwinCriticState
    lambda1_critic: TwinCriticState | None
    lambda2_critic: TwinCriticState | None
    feature_extractor_target: nnx.GraphState
    actor_target: nnx.GraphState
    critic_target: TwinCriticState
    lambda1_critic_target: TwinCriticState | None
    lambda2_critic_target: TwinCriticState | None
    fe_optimizer_state: optax.OptState
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    lambda1_critic_optimizer_state: optax.OptState | None
    lambda2_critic_optimizer_state: optax.OptState | None
    online_buffer: Buffer
    update_step: jax.Array


class AgentFunctions(NamedTuple):
    """Pure closures. See :func:`create_actor_critic` for the training protocol.

    ``predict(state, obs, carry, key, deterministic=False, return_prob=False,
    prev_action=None)`` returns an action and memory carry (and optionally its
    probability). ``train_unrolled`` threads both memory and previous action
    explicitly, so callers can retain them between training rounds.
    """

    predict: Callable
    prefill_buffer: Callable
    train_unrolled: Callable
    encode_action: Callable


class DebugFunctions(NamedTuple):
    """Read-only network diagnostics and a single update for regression tests."""

    get_importance_ratios: Callable
    calculate_latent: Callable
    predict_qpi: Callable
    loss: Callable
    update_step: Callable


class Hyperparameters(NamedTuple):
    """Paper training settings, with configurable learning rates and horizons.

    A replay window contains ``burn_in_length + sequence_length +
    lambda_truncation`` observations. Burn-in starts at zero and is detached.
    The auxiliary losses exclude the look-ahead tail. As in the reported runs,
    the actor uses the full post-burn-in window; task regression and discrepancy
    use that window except its last step. The entropy coefficient is fixed.
    """

    fe_lr: float = 1e-4
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    lambda_critic_lr: float = 1e-4
    alpha: float = 0.1
    gamma: float = 0.99
    online_buffer_size: int = 100_000
    batch_size: int = 128
    tau: float = 0.005
    lambda1: float = 0.05
    lambda2: float = 0.85
    burn_in_length: int = 10
    sequence_length: int = 40
    lambda_truncation: int = 50
    lambda_coef: float = 0.01


def retrace_targets(q_taken, v, rewards, dones, ratios, gamma, lam):
    """Truncated Retrace(λ) Q-targets for time-major ``(T, B)`` arrays.

    ``delta_t = r_t + gamma * (1-done_t) * V_{t+1} - Q_t`` and
    ``acc_t = delta_t + gamma * (1-done_t) * c_{t+1} * acc_{t+1}``,
    where ``c_t = lam * min(1, pi(a_t|s_t) / mu(a_t|s_t))``.
    The current residual always has coefficient one. The terminal carry and
    unavailable final bootstrap are zero; callers exclude a look-ahead tail
    from regression. Episode boundaries stop both bootstrapping and traces.
    """
    v_next = jnp.concatenate([v[1:], jnp.zeros_like(v[:1])], axis=0)
    deltas = rewards + (1.0 - dones) * gamma * v_next - q_taken

    def scan_fn(carry, x):
        acc_next, c_next = carry
        delta, done, ratio = x
        acc = delta + (1.0 - done) * gamma * c_next * acc_next
        return (acc, lam * jnp.minimum(1.0, ratio)), acc

    _, accs = jax.lax.scan(
        scan_fn, (jnp.zeros_like(deltas[0]), jnp.zeros_like(deltas[0])),
        (deltas, dones, ratios), reverse=True, unroll=8,
    )
    return q_taken + accs


def create_actor_critic(
    params: Hyperparameters,
    obs_shape: tuple[int, ...],
    action_dim: int,
    feature_extractor: nnx.Module,
    key: jax.Array,
    *,
    obs_key="observations",
    action_key="actions",
    train_steps=1000,
    actor_dims=(),
    critic_dims=(256, 256),
    lambda1_critic_dims=(256, 256),
    lambda2_critic_dims=(256, 256),
    approximate_lambda=True,
    use_prev_action=False,
    critic_layer_norm=False,
    obs_fn=lambda obs: obs,
    mask_fn=None,
    debug=False,
):
    """Build a discrete actor–critic and its pure training functions.

    ``obs_shape`` describes the full observation stored in replay. ``obs_fn``
    selects the network input; ``mask_fn`` optionally selects legal actions from
    the full observation. Both must support arbitrary leading batch dimensions.
    The feature extractor implements ``(carry, obs, prev_action) -> (carry, z)``.

    Set ``approximate_lambda=False`` for the actor–critic baseline; setting
    ``lambda_coef=0`` instead retains both auxiliary regression losses.
    ``train_unrolled(state, env, env_params, env_state, carry, prev_action, key)``
    returns ``(state, env_state, carry, prev_action, metrics)`` after
    ``train_steps`` collection/update steps. Prefill replay first with
    ``prefill_buffer(state, env, env_params, env_state, n_steps, key)``.
    Gymnax auto-reset handles episode boundaries.
    """
    window = params.burn_in_length + params.sequence_length + params.lambda_truncation
    if action_dim < 1 or train_steps < 1 or params.batch_size < 1:
        raise ValueError("action_dim, train_steps and batch_size must be positive")
    if params.burn_in_length < 0 or params.sequence_length < 1 or params.lambda_truncation < 1:
        raise ValueError("require nonnegative burn-in and positive sequence and tail lengths")
    if params.online_buffer_size <= window:
        raise ValueError("online_buffer_size must exceed the replay window length")
    if not 0 <= params.lambda1 < params.lambda2 <= 1:
        raise ValueError("require 0 <= lambda1 < lambda2 <= 1")
    if not 0 < params.gamma < 1 or not 0 < params.tau <= 1:
        raise ValueError("require 0 < gamma < 1 and 0 < tau <= 1")
    if min(params.alpha, params.lambda_coef, params.fe_lr, params.actor_lr,
           params.critic_lr, params.lambda_critic_lr) < 0:
        raise ValueError("loss coefficients and learning rates must be nonnegative")
    expected_pa = action_dim if use_prev_action else 0
    if getattr(feature_extractor, "prev_action_dim", 0) != expected_pa:
        raise ValueError(f"feature extractor prev_action_dim must equal {expected_pa}")

    def encode_action(action):
        idx = jnp.round(action).astype(jnp.int32)
        if idx.ndim and idx.shape[-1] == 1:
            idx = idx[..., 0]
        return jax.nn.one_hot(idx, action_dim, dtype=jnp.float32)

    shapes = {obs_key: tuple(obs_shape), action_key: (1,), reward_key: (),
              terminated_key: (), behaviour_key: ()}
    buffer, buffer_fns = create_buffer(
        shapes, params.online_buffer_size, params.batch_size,
        list(shapes), [obs_key],
    )
    buffer = buffer._replace(info={**buffer.info,
                                  behaviour_key: jnp.ones_like(buffer.info[behaviour_key])})
    sample_sequences = create_sequence_sample(
        buffer.size, params.batch_size, window, list(shapes))
    _, dummy = feature_extractor(
        feature_extractor.initialize_carry(1), obs_fn(jnp.zeros((1, *obs_shape))))
    feature_dim = dummy.shape[-1]
    keys = jax.random.split(key, 7)

    def make_head(dims, head_key, layer_norm=False):
        return nnx.split(Head(feature_dim, dims, action_dim,
                              layer_norm=layer_norm, rngs=nnx.Rngs(head_key)))

    def make_twin(dims, k1, k2):
        g1, s1 = make_head(dims, k1, critic_layer_norm)
        g2, s2 = make_head(dims, k2, critic_layer_norm)
        return TwinCriticGraph(g1, g2), TwinCriticState(s1, s2)

    fe_graph, fe_state = nnx.split(feature_extractor)
    actor_graph, actor_state = make_head(actor_dims, keys[0])
    critic_graph, critic_state = make_twin(critic_dims, keys[1], keys[2])
    l1_graph = l2_graph = l1_state = l2_state = None
    if approximate_lambda:
        l1_graph, l1_state = make_twin(lambda1_critic_dims, keys[3], keys[4])
        l2_graph, l2_state = make_twin(lambda2_critic_dims, keys[5], keys[6])
    optimizers = tuple(optax.adam(lr) for lr in (
        params.fe_lr, params.actor_lr, params.critic_lr,
        params.lambda_critic_lr, params.lambda_critic_lr))
    networks = (fe_state, actor_state, critic_state, l1_state, l2_state)
    opt_states = tuple(opt.init(s) if s is not None else None
                       for opt, s in zip(optimizers, networks))
    state = AgentState(*networks, *networks, *opt_states, buffer, jnp.int32(0))
    state = jax.tree.map(
        lambda x: jnp.asarray(x, dtype=x.dtype) if hasattr(x, "dtype") else x, state)

    def masked_logits(actor, x, mask=None):
        logits = nnx.merge(actor_graph, actor)(x)
        if mask is None:
            return logits
        mask = mask.astype(bool)
        # Zero-padded rows may have no legal action. Keep their softmax finite.
        mask = mask | ~jnp.any(mask, axis=-1, keepdims=True)
        return jnp.where(mask, logits, -1e9)

    def policy(actor, x, mask=None):
        logits = masked_logits(actor, x, mask)
        return jax.nn.softmax(logits), jax.nn.log_softmax(logits)

    def q_values(critic, graph, x):
        return jnp.stack([nnx.merge(graph.q1, critic.q1)(x),
                          nnx.merge(graph.q2, critic.q2)(x)], axis=-1)

    def q_both(critic, graph, x, actions):
        q = q_values(critic, graph, x)
        idx = jnp.round(actions.reshape(-1)).astype(jnp.int32)
        return q[jnp.arange(q.shape[0]), idx, 0], q[jnp.arange(q.shape[0]), idx, 1]

    def q_min(critic, graph, x, actions):
        return jnp.minimum(*q_both(critic, graph, x, actions))

    def value(actor, critic, graph, x, mask=None):
        probs, _ = policy(actor, x, mask)
        return (probs * jnp.min(q_values(critic, graph, x), axis=-1)).sum(-1)

    @jax.jit
    def get_importance_ratios(actor, x, actions, behaviour_probs, mask=None):
        probs, _ = policy(actor, x, mask)
        idx = jnp.round(actions).astype(jnp.int32).reshape(probs.shape[:-1] + (1,))
        return jnp.take_along_axis(probs, idx, axis=-1).squeeze(-1) / behaviour_probs

    def features(agent, obs, carry, prev_action):
        pa = prev_action[None] if prev_action is not None else None
        new_carry, x = nnx.merge(fe_graph, agent.feature_extractor)(
            carry[None], obs_fn(obs)[None], pa)
        return new_carry[0], x

    @partial(jax.jit, static_argnames=["deterministic", "return_prob"])
    def predict(agent, obs, carry, key, deterministic=False, return_prob=False,
                prev_action=None):
        new_carry, x = features(agent, obs, carry, prev_action)
        mask = mask_fn(obs) if mask_fn is not None else None
        logits = masked_logits(agent.actor, x, mask)[0]
        action = jnp.argmax(logits) if deterministic else jax.random.categorical(key, logits)
        result = (action.astype(jnp.float32), new_carry)
        return (*result, jax.nn.softmax(logits)[action]) if return_prob else result

    @jax.jit
    def predict_qpi(agent, obs, carry, prev_action=None):
        new_carry, x = features(agent, obs, carry, prev_action)
        mask = mask_fn(obs) if mask_fn is not None else None
        probs, _ = policy(agent.actor, x, mask)
        return jnp.min(q_values(agent.critic, critic_graph, x), -1)[0], probs[0], new_carry

    def calculate_latent(fe, target_fe, observations, actions, dones, init_carries):
        """Zero-start replay unroll with detached burn-in and episode resets."""
        model, target_model = nnx.merge(fe_graph, fe), nnx.merge(fe_graph, target_fe)
        obs = jnp.swapaxes(obs_fn(observations), 0, 1)
        dones = jnp.swapaxes(dones, 0, 1)
        enc = (encode_action(jnp.swapaxes(actions, 0, 1)) if use_prev_action else
               jnp.zeros((*obs.shape[:2], 0), jnp.float32))

        def step(carries, inputs):
            carry, target_carry, pa = carries
            observation, done, action = inputs
            c, y = model(carry, observation, pa if use_prev_action else None)
            tc, ty = target_model(target_carry, observation, pa if use_prev_action else None)
            reset = done.astype(bool)[:, None]
            return (jnp.where(reset, 0, c), jnp.where(reset, 0, tc),
                    jnp.where(reset, 0, action)), (y, ty)

        initial = (init_carries, init_carries, jnp.zeros_like(enc[0]))
        bl = params.burn_in_length
        burnt, _ = jax.lax.scan(step, initial, (obs[:bl], dones[:bl], enc[:bl]), unroll=8)
        _, latent = jax.lax.scan(step, jax.lax.stop_gradient(burnt),
                                 (obs[bl:], dones[bl:], enc[bl:]), unroll=8)
        return latent

    def flat(x):
        return x.reshape((-1, *x.shape[2:]))

    def auxiliary_loss(q, tq, graph, lam, target_actor, latent, target_latent,
                       actions, rewards, dones, ratios, masks):
        shape = rewards.shape
        v = value(target_actor, tq, graph, flat(target_latent), masks).reshape(shape)
        target_taken = q_min(tq, graph, flat(target_latent), flat(actions)).reshape(shape)
        targets = retrace_targets(target_taken, v, rewards, dones, ratios, params.gamma, lam)
        q1, q2 = (a.reshape(shape) for a in q_both(q, graph, flat(latent), flat(actions)))
        end = -params.lambda_truncation
        targets = jax.lax.stop_gradient(targets)
        # Both twins regress independently. The minimum belongs only in targets.
        loss = 0.5 * (optax.huber_loss(q1[:end], targets[:end]).mean()
                      + optax.huber_loss(q2[:end], targets[:end]).mean())
        return loss, {
            f"lambda{lam}_loss": loss,
            f"lambda{lam}_critic": jnp.minimum(q1, q2)[:end].mean(),
            f"lambda{lam}_target": targets[:end].mean(),
            f"lambda{lam}_twin_gap": jnp.abs(q1 - q2).mean(),
        }

    def loss_combined(fe, actor, critic, l1, l2, agent, key):
        # Keep the recorded runs' sampling PRNG split, even though exact action
        # expectations require no random keys in the individual loss terms.
        key_sample = jax.random.split(key, 5)[0]
        sample, _ = sample_sequences(agent.online_buffer, key_sample)
        data = sample.this_info
        latent, target_latent = calculate_latent(
            fe, agent.feature_extractor_target, data[obs_key], data[action_key],
            data[terminated_key], feature_extractor.initialize_carry(params.batch_size))
        bl = params.burn_in_length
        actions, rewards, dones, behaviour = (
            jnp.swapaxes(data[k], 0, 1)[bl:]
            for k in (action_key, reward_key, terminated_key, behaviour_key))
        masks = (jnp.swapaxes(mask_fn(data[obs_key]), 0, 1)[bl:]
                 if mask_fn is not None else None)
        masks_flat = flat(masks) if masks is not None else None
        actor_latent = (jax.lax.stop_gradient(flat(latent))
                        if feature_extractor.carry_dim else flat(latent))
        probs, log_probs = policy(actor, actor_latent, masks_flat)
        actor_q = jnp.min(q_values(jax.lax.stop_gradient(critic), critic_graph, actor_latent), -1)
        entropy = -(probs * log_probs).sum(-1)
        expected_q = (probs * actor_q).sum(-1)
        v = expected_q + params.alpha * entropy
        actor_loss = -v.mean()
        next_v = value(agent.actor_target, agent.critic_target, critic_graph,
                       flat(target_latent[1:]), flat(masks[1:]) if masks is not None else None)
        target = jax.lax.stop_gradient(
            rewards[:-1].reshape(-1) + params.gamma * (1 - dones[:-1].reshape(-1)) * next_v)
        q1, q2 = q_both(critic, critic_graph, flat(latent[:-1]), flat(actions[:-1]))
        critic_loss = 0.5 * (((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean())
        loss = actor_loss + critic_loss
        metrics = {"q": expected_q.mean(), "entropy": entropy.mean(), "v": v.mean(),
                   "actor_loss": actor_loss, "critic_loss": critic_loss,
                   "target_q": target.mean(), "alpha": jnp.asarray(params.alpha)}
        if approximate_lambda:
            ratios = get_importance_ratios(
                agent.actor_target, flat(target_latent), flat(actions),
                behaviour.reshape(-1), masks_flat).reshape(behaviour.shape)
            for q, tq, graph, lam in (
                (l1, agent.lambda1_critic_target, l1_graph, params.lambda1),
                (l2, agent.lambda2_critic_target, l2_graph, params.lambda2),
            ):
                aux_loss, aux_metrics = auxiliary_loss(
                    q, tq, graph, lam, agent.actor_target, latent, target_latent,
                    actions, rewards, dones, ratios, masks_flat)
                loss += aux_loss
                metrics.update(aux_metrics)
            d = (q_min(jax.lax.stop_gradient(l1), l1_graph, flat(latent[:-1]), flat(actions[:-1]))
                 - q_min(jax.lax.stop_gradient(l2), l2_graph, flat(latent[:-1]), flat(actions[:-1])))
            ld = (d ** 2).mean()
            metrics.update(ld_loss=ld, ld_mean=d.mean(), ld_std=d.std())
            loss += params.lambda_coef * ld
        return loss, metrics

    @jax.jit
    def update_step(agent, key):
        networks = agent[:5]
        grads, metrics = jax.grad(loss_combined, argnums=(0, 1, 2, 3, 4), has_aux=True)(
            *networks, agent, key)
        new_networks, new_opts = [], []
        for opt, grad, network, opt_state in zip(optimizers, grads, networks, agent[10:15]):
            if network is None:
                new_networks.append(None)
                new_opts.append(None)
            else:
                updates, new_opt = opt.update(grad, opt_state)
                new_networks.append(optax.apply_updates(network, updates))
                new_opts.append(new_opt)
        targets = jax.tree.map(lambda old, new: (1 - params.tau) * old + params.tau * new,
                               tuple(agent[5:10]), tuple(new_networks))
        return AgentState(*new_networks, *targets, *new_opts,
                          agent.online_buffer, agent.update_step + 1), metrics

    def train_unrolled(agent, env, env_params, env_state, carry, prev_action, key):
        """Collect and update; thread rollout history across calls explicitly."""
        def step(scan_carry, _):
            agent, es, memory, pa, key = scan_carry
            _, next_key, env_key, update_key = jax.random.split(key, 4)
            ak, ek = jax.random.split(env_key)
            obs = env.get_obs(es, env_params)
            action, memory, prob = predict(agent, obs, memory, ak, return_prob=True,
                                           prev_action=pa if use_prev_action else None)
            _, es, reward, done, _ = env.step(ek, es, action.astype(jnp.int32), env_params)
            transition = {obs_key: obs, action_key: jnp.atleast_1d(action),
                          reward_key: jnp.asarray(reward, jnp.float32),
                          terminated_key: jnp.asarray(done, jnp.float32), behaviour_key: prob}
            agent = agent._replace(online_buffer=buffer_fns.add(agent.online_buffer, transition, done))
            memory = jnp.where(done, jnp.zeros_like(memory), memory)
            pa = encode_action(jnp.atleast_1d(action)) if use_prev_action else pa
            pa = jnp.where(done, jnp.zeros_like(pa), pa)
            agent, metrics = update_step(agent, update_key)
            return (agent, es, memory, pa, next_key), metrics

        (agent, es, carry, pa, _), metrics = jax.lax.scan(
            step, (agent, env_state, carry, prev_action, key), length=train_steps)
        return agent, es, carry, pa, jax.tree.map(lambda x: x.mean(), metrics)

    @partial(jax.jit, static_argnames=["env", "n_steps"])
    def prefill_buffer(agent, env, env_params, env_state, n_steps, key):
        """Uniform random replay prefill (uniform over legal actions if masked).

        The last transition is marked terminal, matching the reported prefill
        protocol. Callers reset the environment before policy-driven training.
        """
        if not window < n_steps <= params.online_buffer_size:
            raise ValueError("prefill steps must exceed the replay window and fit the buffer")

        def step(carry, idx):
            agent, es, key = carry
            key, ak, ek = jax.random.split(key, 3)
            obs = env.get_obs(es, env_params)
            if mask_fn is None:
                action = jax.random.randint(ak, (), 0, action_dim)
                prob = jnp.float32(1 / action_dim)
            else:
                legal = mask_fn(obs).astype(bool)
                legal = legal | ~jnp.any(legal)
                action = jax.random.categorical(ak, jnp.where(legal, 0.0, -1e9))
                prob = 1 / legal.sum().astype(jnp.float32)
            _, es, reward, done, _ = env.step(ek, es, action, env_params)
            terminated = done | (idx == n_steps - 1)
            transition = {obs_key: obs, action_key: jnp.atleast_1d(action).astype(jnp.float32),
                          reward_key: jnp.asarray(reward, jnp.float32),
                          terminated_key: jnp.asarray(terminated, jnp.float32), behaviour_key: prob}
            agent = agent._replace(online_buffer=buffer_fns.add(
                agent.online_buffer, transition, terminated))
            return (agent, es, key), None

        (agent, es, _), _ = jax.lax.scan(step, (agent, env_state, key), jnp.arange(n_steps))
        return agent, es

    fns = AgentFunctions(predict, prefill_buffer, train_unrolled, encode_action)
    if debug:
        return state, fns, DebugFunctions(get_importance_ratios, calculate_latent,
                                          predict_qpi, loss_combined, update_step)
    return state, fns
