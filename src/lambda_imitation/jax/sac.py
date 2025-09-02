from functools import partial
from typing import Any, Callable, NamedTuple, Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from lambda_imitation.jax.buffer import (Buffer, BufferFunctions, BufferSample,
                                         create_buffer)

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class Hyperparameters(NamedTuple):
    seed: int = 42
    """The seed used"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    use_q_targets: bool = True
    """Whether or not to use target nets for the Q function"""
    use_policy_targets: bool = True
    """Whether or not to use target nets for the policy"""
    policy_update_frequency: int = 1
    """How often targets are updated"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    learning_starts: int = 0
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    feature_extractor_lr: float = 3e-4
    """the learning rate of the feature extractor network optimizer"""
    alpha: float = 1.0
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy: float = 0.2
    """The target entropy when not chosen automatically"""
    hidden_state_dim: int = 10
    """Dimension of hidden states, has to be even, if 0, no LSTM will be used"""
    hidden_state_recalculation_interval: int = 500
    """How often the hidden states of the demonstration buffer are recalculated. Note that logging frequency must be a multiple of this number, or else training steps might differ for technical reasons."""
    recalculate_hidden_states_in_update: bool = False
    """Whether or not to recalculate hidden states at every step for sample"""
    lambda_discrepancy_coef: float = 0.4
    """Coefficient for lambda discrepancy loss. If 0, no MC approximation will be calculated."""
    use_action_recalculation: bool = False
    """Whether or not to re-calculate actions in calculation of lambda-discrepancy"""
    resample_actions: bool = True
    """Whether or not to resample actions or just take buffer actions when calculating lambda discrepancy"""


class Network(NamedTuple):
    net: nnx.Module
    target_net: nnx.Module | None
    optimizer: nnx.Optimizer


class Alpha(NamedTuple):
    log_alpha: jax.Array
    alpha: jax.Array
    optimizer_update: Any
    optimizer_state: Any


def create_alpha(start_alpha, lr):
    start_alpha = jnp.array(start_alpha, dtype=jnp.float32)
    log_alpha = jnp.log(start_alpha)
    optimizer = optax.adam(lr)
    init_state = optimizer.init(log_alpha)
    return Alpha(
        log_alpha, start_alpha, jax.tree_util.Partial(optimizer.update), init_state
    )


def create_network(net, target_net, lr):
    return Network(
        net=net, target_net=target_net, optimizer=nnx.Optimizer(net, optax.adam(lr), wrt=nnx.Param)  # type: ignore
    )


class SACState(NamedTuple):
    feature_extractor: Network
    actor: Network
    q1: Network
    q2: Network
    mc_q1: Network
    mc_q2: Network
    alpha: Alpha
    buffer: Buffer
    obs: jax.Array
    hidden_state: jax.Array
    env_state: Any
    random_key: Any
    n_updates: int


class SACFunctions(NamedTuple):
    learn: Callable
    predict: Callable


@partial(nnx.jit, static_argnums=[0, 1, 7, 9])
def run_env_step(
    env,
    env_params,
    action,
    buffer: Buffer,
    current_obs,
    hidden_state,
    env_state,
    buffer_functions,
    gamma,
    calculate_return,
    key,
) -> Tuple[Buffer, jax.Array, jax.Array]:
    key_step, key_reset = jax.random.split(key)
    obs, state, reward, done, _ = env.step(key_step, env_state, action, env_params)
    new_buffer: Buffer = buffer_functions.add(
        buffer, current_obs, hidden_state, action, reward, done, gamma, calculate_return
    )
    # reset_obs, reset_state = env.reset(key_reset, env_params)
    # return_obs = jax.tree.map(lambda x, y: done * x + (1 - done) * y, reset_obs, obs)
    # return_state = jax.tree.map(
    #     lambda x, y: done * x + (1 - done) * y, reset_state, state
    # )
    return new_buffer, obs, done, state  # type: ignore


def create_SAC(
    env,
    env_params,
    observation_space_size,
    discrete_action_space: bool,
    action_space_size,
    learn_steps: int,
    params: Hyperparameters,
    action_scale: float = 0.0,
    action_bias: float = 0.0,
):
    def get_action(observations, actor_net, key):
        if discrete_action_space:
            logits = actor_net(observations)
            actions = jax.random.categorical(key, logits)
            entropy = jax.nn.softmax(logits, axis=1) * jax.nn.log_softmax(
                logits, axis=1
            )
            return actions, entropy.sum(axis=1)
        else:
            actor_out = actor_net(observations)  # type: ignore
            mean, log_std = (
                actor_out[..., :action_space_size],
                actor_out[..., action_space_size:],
            )
            log_std = jnp.tanh(log_std)
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
            std = jnp.exp(log_std)
            x_t = jax.random.normal(key, mean.shape) * std + mean
            y_t = jnp.tanh(x_t)
            action = y_t * action_scale + action_bias
            log_prob = (
                -((x_t - mean) ** 2) / (2 * std)
                - jnp.sqrt(2 * jnp.pi * std**2)
                - jnp.log(action_scale * (1 - y_t**2) + 1e-6)
            )
            return action, log_prob

    def get_q_values(observations, actions, net):
        if discrete_action_space:
            return jnp.take_along_axis(
                net(observations),
                jnp.reshape(actions.astype(jnp.int32), (params.batch_size, 1)),
                1,
            ).reshape(-1)
        else:
            return net(jnp.concatenate([observations, actions], axis=-1)).reshape(-1)

    def loss_q(
        q1_net: nnx.Module,
        q2_net: nnx.Module,
        feature_extractor_net: nnx.Module,
        q1_target_net: nnx.Module,
        q2_target_net: nnx.Module,
        feature_extractor_target_net: nnx.Module,
        actor_net: nnx.Module,
        alpha: float,
        buffer: Buffer,
        key,
    ):
        key_sample, key_next_actions = jax.random.split(key)
        sample: BufferSample = buffer_functions.sample(buffer, key_sample)
        feature_obs, hidden_state_obs = feature_extractor_net(  # type: ignore
            sample.hidden_states, sample.observations
        )
        next_feature_obs, _ = (feature_extractor_target_net if params.use_q_targets else feature_extractor_net)(  # type: ignore
            sample.next_hidden_states, sample.next_observations
        )
        if discrete_action_space:
            logits = actor_net(next_feature_obs)  # type: ignore
            probs = jax.nn.softmax(logits, axis=1)
            log_prob = jax.nn.log_softmax(logits, axis=1)
            qf1_next_target = (q1_target_net if params.use_q_targets else q1_net)(  # type: ignore
                next_feature_obs
            )
            qf2_next_target = (q2_target_net if params.use_q_targets else q2_net)(  # type: ignore
                next_feature_obs
            )
            min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target)
            target_q_values = sample.rewards + (
                1 - sample.terminated
            ) * params.gamma * ((min_qf_next_target - alpha * log_prob) * probs).sum(-1)
            target_q_values = jax.lax.stop_gradient(target_q_values)
        else:
            actor_out = actor_net(next_feature_obs)  # type: ignore
            mean, log_std = (
                actor_out[..., :action_space_size],
                actor_out[..., action_space_size:],
            )
            log_std = jnp.tanh(log_std)
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
            std = jnp.exp(log_std)
            x_t = jax.random.normal(key_next_actions, mean.shape) * std + mean
            y_t = jnp.tanh(x_t)
            next_actions = y_t * action_scale + action_bias
            log_prob = (
                -((x_t - mean) ** 2) / (2 * std)
                - jnp.sqrt(2 * jnp.pi * std**2)
                - jnp.log(action_scale * (1 - y_t**2) + 1e-6)
            )
            qf1_next_target = get_q_values(
                next_feature_obs,
                next_actions,
                q1_target_net if params.use_q_targets else q1_net,
            )
            qf2_next_target = get_q_values(
                next_feature_obs,
                next_actions,
                q2_target_net if params.use_q_targets else q2_net,
            )
            min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target)
            target_q_values = sample.rewards + (
                1 - sample.terminated
            ) * params.gamma * (min_qf_next_target - alpha * log_prob)
            target_q_values = jax.lax.stop_gradient(target_q_values)

        qf1_values = get_q_values(feature_obs, sample.actions, q1_net)
        qf2_values = get_q_values(feature_obs, sample.actions, q2_net)

        q1_loss = (qf1_values - target_q_values) ** 2
        q2_loss = (qf2_values - target_q_values) ** 2
        return (q1_loss + q2_loss).mean(), (qf1_values, qf2_values)

    def loss_mc(
        mc_q1_net: nnx.Module,
        mc_q2_net: nnx.Module,
        feature_extractor_net: nnx.Module,
        buffer: Buffer,
        key,
    ):
        sample: BufferSample = buffer_functions.sample(buffer, key)
        feature_obs, _ = feature_extractor_net(  # type: ignore
            sample.hidden_states, sample.observations
        )
        mc_q1 = get_q_values(feature_obs, sample.actions, mc_q1_net)
        mc_q2 = get_q_values(feature_obs, sample.actions, mc_q2_net)

        q1_loss = ((mc_q1 - sample.returns) ** 2).mean()
        q2_loss = ((mc_q2 - sample.returns) ** 2).mean()
        q_loss = q1_loss + q2_loss
        return q_loss

    def loss_alpha(
        log_alpha: jax.Array,
        actor_net: nnx.Module,
        feature_extractor_net: nnx.Module,
        buffer: Buffer,
        key,
    ):
        sample: BufferSample = buffer_functions.sample(buffer, key)
        feature_obs, _ = feature_extractor_net(  # type: ignore
            sample.hidden_states, sample.observations
        )
        if discrete_action_space:
            logits = actor_net(feature_obs)  # type: ignore
            action_probs = jax.nn.softmax(logits, axis=1)
            log_prob = jax.nn.log_softmax(logits, axis=1)
            entropy = (log_prob * action_probs).sum(axis=1).mean()
            alpha_loss = -jnp.exp(log_alpha) * (entropy + params.target_entropy)
            return alpha_loss, entropy
        else:
            _, log_pi = get_action(feature_obs, actor_net, key)
            alpha_loss = -jnp.exp(log_alpha) * (log_pi.mean() + params.target_entropy)
            return alpha_loss, log_pi.mean()

    def loss_actor(
        actor_net: nnx.Module,
        q1_net: nnx.Module,
        q2_net: nnx.Module,
        feature_extractor_net: nnx.Module,
        alpha: float,
        buffer: Buffer,
        key,
    ):
        sample: BufferSample = buffer_functions.sample(buffer, key)
        feature_obs, _ = feature_extractor_net(  # type: ignore
            sample.hidden_states, sample.observations
        )
        if discrete_action_space:
            qf1_pi = q1_net(feature_obs)  # type: ignore
            qf2_pi = q2_net(feature_obs)  # type: ignore
            min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)

            logits = actor_net(feature_obs)  # type: ignore
            action_probs = jax.nn.softmax(logits, axis=1)
            log_prob = jax.nn.log_softmax(logits, axis=1)
            return (((alpha * log_prob) - min_qf_pi) * action_probs).sum(axis=-1).mean()
        else:
            pi, log_pi = get_action(feature_obs, actor_net, key)
            xa = jnp.concatenate([feature_obs, pi], axis=-1)
            qf1_pi = q1_net(xa)  # type: ignore
            qf2_pi = q2_net(xa)  # type: ignore
            min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)
            return ((alpha * log_pi) - min_qf_pi).mean()

    def loss_lambda_discrepancy(
        feature_extractor_net: nnx.Module,
        mc_q1_net: nnx.Module,
        mc_q2_net: nnx.Module,
        q1_net: nnx.Module,
        q2_net: nnx.Module,
        actor_net: nnx.Module,
        buffer: Buffer,
        key,
    ):
        sample: BufferSample = buffer_functions.sample(buffer, key)
        feature_obs, _ = feature_extractor_net(  # type: ignore
            sample.hidden_states, sample.observations
        )
        actions = sample.actions
        if params.resample_actions:
            pi, _ = get_action(feature_obs, actor_net, key)
            actions = pi
        mc_q1 = get_q_values(feature_obs, actions, mc_q1_net)
        mc_q2 = get_q_values(feature_obs, actions, mc_q2_net)
        mc_q = jnp.minimum(mc_q1, mc_q2)

        q1 = get_q_values(feature_obs, actions, q1_net)
        q2 = get_q_values(feature_obs, actions, q2_net)
        q = jnp.minimum(q1, q2)

        return ((mc_q - q) ** 2).mean(), (mc_q, q)

    @nnx.jit
    def learn(
        state: SACState,
    ):
        print(state.random_key)

        def update_step(state: SACState, _unused_scan_input):
            (
                key_next,
                key_q,
                key_actor,
                key_action,
                key_env,
                key_alpha,
                key_mc,
                key_ld,
            ) = jax.random.split(state.random_key, 8)
            action, hidden_state = predict(
                state.obs,
                state.hidden_state,
                state.actor.net,
                state.feature_extractor.net,
                key_action,
            )
            buffer, obs, done, env_state = run_env_step(
                env,
                env_params,
                action,
                state.buffer,
                state.obs,
                state.hidden_state,
                state.env_state,
                buffer_functions,
                params.gamma,
                params.lambda_discrepancy_coef > 0.0,
                key_env,
            )
            hidden_state = jax.lax.select(
                done, jnp.zeros((2 * params.hidden_state_dim,)), hidden_state[0]
            )

            (value_loss_q, (q1_values, q2_values)), grads_q = nnx.value_and_grad(
                loss_q, has_aux=True, argnums=[0, 1, 2]
            )(
                state.q1.net,
                state.q2.net,
                state.feature_extractor.net,
                state.q1.target_net,
                state.q2.target_net,
                state.feature_extractor.target_net,
                state.actor.net,
                state.alpha.alpha,
                buffer,
                key_q,
            )

            value_loss_actor, grads_actor = nnx.value_and_grad(loss_actor)(
                state.actor.net,
                state.q1.net,
                state.q2.net,
                state.feature_extractor.net,
                state.alpha.alpha,
                buffer,
                key_actor,
            )
            grads_fe = grads_q[2]

            if params.lambda_discrepancy_coef > 0:
                value_loss_mc, grads_mc = nnx.value_and_grad(
                    loss_mc, has_aux=False, argnums=[0, 1]
                )(
                    state.mc_q1.net,
                    state.mc_q2.net,
                    state.feature_extractor.net,
                    buffer,
                    key_mc,
                )

                (value_loss_ld, (mc_q, q)), grads_ld = nnx.value_and_grad(
                    loss_lambda_discrepancy, has_aux=True
                )(
                    state.feature_extractor.net,
                    state.mc_q1.target_net if params.use_q_targets else state.mc_q1.net,
                    state.mc_q2.target_net if params.use_q_targets else state.mc_q2.net,
                    state.q1.target_net if params.use_q_targets else state.q1.net,
                    state.q2.target_net if params.use_q_targets else state.q2.net,
                    state.actor.net,
                    buffer,
                    key_ld,
                )

            state.actor.optimizer.update(state.actor.net, grads_actor)
            state.q1.optimizer.update(state.q1.net, grads_q[0])
            state.q2.optimizer.update(state.q2.net, grads_q[1])
            if params.lambda_discrepancy_coef > 0:
                state.mc_q1.optimizer.update(state.mc_q1.net, grads_mc[0])
                state.mc_q2.optimizer.update(state.mc_q2.net, grads_mc[1])
                grads_fe = jax.tree.map(lambda x, y: (1-params.lambda_discrepancy_coef)*x + params.lambda_discrepancy_coef * y, grads_fe, grads_ld)
            state.feature_extractor.optimizer.update(
                state.feature_extractor.net, grads_q[2]
            )

            if params.autotune:
                (value_loss_alpha, entropy), grads_alpha = nnx.value_and_grad(
                    loss_alpha, has_aux=True
                )(
                    state.alpha.log_alpha,
                    state.actor.net,
                    state.feature_extractor.net,
                    buffer,
                    key_alpha,
                )
                updates, new_opt_state = state.alpha.optimizer_update(
                    grads_alpha, state.alpha.optimizer_state, state.alpha.log_alpha
                )
                new_log_alpha = optax.apply_updates(state.alpha.log_alpha, updates)
                new_alpha = Alpha(
                    new_log_alpha,
                    jnp.exp(new_log_alpha),
                    state.alpha.optimizer_update,
                    new_opt_state,
                )
            else:
                new_alpha = state.alpha
                value_loss_alpha = 0
                entropy = 0

            # target update
            new_q1 = state.q1
            new_q2 = state.q2
            new_mc_q1 = state.mc_q1
            new_mc_q2 = state.mc_q2
            new_feature_extractor = state.feature_extractor
            if params.use_q_targets:
                update_param = jax.lax.select(
                    state.n_updates % params.policy_update_frequency == 0,
                    params.tau,
                    0.0,
                )
                new_q1_target = jax.tree.map(
                    lambda net, target: update_param * net
                    + (1 - update_param) * target,
                    state.q1.net,
                    state.q1.target_net,
                )
                new_q1 = Network(
                    net=state.q1.net,
                    target_net=new_q1_target,
                    optimizer=state.q1.optimizer,
                )
                new_q2_target = jax.tree.map(
                    lambda net, target: update_param * net
                    + (1 - update_param) * target,
                    state.q2.net,
                    state.q2.target_net,
                )
                new_q2 = Network(
                    net=state.q2.net,
                    target_net=new_q2_target,
                    optimizer=state.q2.optimizer,
                )
                new_feature_extractor_target = jax.tree.map(
                    lambda net, target: update_param * net
                    + (1 - update_param) * target,
                    state.feature_extractor.net,
                    state.feature_extractor.target_net,
                )
                new_feature_extractor = Network(
                    net=state.feature_extractor.net,
                    target_net=new_feature_extractor_target,
                    optimizer=state.feature_extractor.optimizer,
                )
                if params.lambda_discrepancy_coef > 0.0:
                    new_mc_q1_target = jax.tree.map(
                        lambda net, target: update_param * net
                        + (1 - update_param) * target,
                        state.mc_q1.net,
                        state.mc_q1.target_net,
                    )
                    new_mc_q1 = Network(
                        net=state.mc_q1.net,
                        target_net=new_mc_q1_target,
                        optimizer=state.mc_q1.optimizer,
                    )
                    new_mc_q2_target = jax.tree.map(
                        lambda net, target: update_param * net
                        + (1 - update_param) * target,
                        state.mc_q2.net,
                        state.mc_q2.target_net,
                    )
                    new_mc_q2 = Network(
                        net=state.mc_q2.net,
                        target_net=new_mc_q2_target,
                        optimizer=state.mc_q2.optimizer,
                    )

            return (
                SACState(
                    feature_extractor=new_feature_extractor,
                    actor=state.actor,
                    q1=new_q1,
                    q2=new_q2,
                    mc_q1=new_mc_q1,
                    mc_q2=new_mc_q2,
                    alpha=new_alpha,
                    buffer=buffer,
                    obs=obs,
                    hidden_state=hidden_state,
                    env_state=env_state,
                    random_key=key_next,
                    n_updates=state.n_updates + 1,
                ),
                {
                    "train/loss_q": value_loss_q,
                    "train/q1_values": q1_values.mean(),
                    "train/q2_values": q2_values.mean(),
                    "train/loss_actor": value_loss_actor,
                    "train/loss_alpha": value_loss_alpha,
                    "train/entropy": -entropy,
                    "train/loss_mc": (
                        0 if params.lambda_discrepancy_coef == 0.0 else value_loss_mc
                    ),
                    "train/loss_ld": (
                        0 if params.lambda_discrepancy_coef == 0.0 else value_loss_ld
                    ),
                },
            )

        recalc = (
            params.hidden_state_recalculation_interval
            if params.hidden_state_recalculation_interval > 0
            else learn_steps
        )

        def recalc_scan(carry, _unused):
            state = carry
            state, metrics = jax.lax.scan(update_step, state, None, recalc)
            buffer = buffer_functions.recalculate_hidden_states(
                state.buffer, feature_extractor.net
            )
            state = SACState(
                feature_extractor=state.feature_extractor,
                actor=state.actor,
                q1=state.q1,
                q2=state.q2,
                mc_q1=state.mc_q1,
                mc_q2=state.mc_q2,
                alpha=state.alpha,
                buffer=buffer,
                obs=state.obs,
                hidden_state=state.hidden_state,
                env_state=state.env_state,
                random_key=state.random_key,
                n_updates=state.n_updates,
            )
            return state, metrics

        state, metrics = jax.lax.scan(
            recalc_scan,
            state,
            None,
            learn_steps // recalc,
        )
        return state, metrics

    def predict(
        obs, hidden_state, actor_net, feature_net, key, deterministic=False
    ) -> Tuple[jax.Array, jax.Array]:
        obs = obs.reshape(1, -1)
        hidden_state = hidden_state.reshape(1, -1)
        feature_obs, hidden_state = feature_net(hidden_state, obs)

        if discrete_action_space:
            logits = actor_net(feature_obs)
            if deterministic:
                return jnp.argmax(logits), hidden_state
            else:
                return jax.random.categorical(key, logits)[0], hidden_state
        else:
            actor_out = actor_net(feature_obs)
            mean, log_std = (
                actor_out[..., :action_space_size],
                actor_out[..., action_space_size:],
            )
            if deterministic:
                return mean.reshape(-1), hidden_state
            log_std = jnp.tanh(log_std)
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
            std = jnp.exp(log_std)
            x_t = jax.random.normal(key, mean.shape) * std + mean
            y_t = jnp.tanh(x_t)
            action = y_t * action_scale + action_bias
            return action.reshape(-1), hidden_state

    class SimpleNet(nnx.Module):
        def __init__(self, din, dout, rngs):
            self.fc1 = nnx.Linear(din, 64, rngs=rngs)
            self.fc2 = nnx.Linear(64, 64, rngs=rngs)
            self.fc3 = nnx.Linear(64, 64, rngs=rngs)
            self.fc4 = nnx.Linear(64, dout, rngs=rngs)

        def __call__(self, x):
            x = nnx.relu(self.fc1(x))
            x = nnx.relu(self.fc2(x))
            x = nnx.relu(self.fc3(x))
            return self.fc4(x)

    class IdentityExtractor(nnx.Module):
        def __init__(self):
            pass

        def __call__(self, carry, x):
            return x, carry

    class LSTMExtractor(nnx.Module):
        def __init__(self, din, dout, rngs):
            self.lstm = nnx.OptimizedLSTMCell(din, dout, rngs=rngs)

        def __call__(self, carry, x):
            carry = (
                carry[..., : params.hidden_state_dim],
                carry[..., params.hidden_state_dim :],
            )
            (c, h), _ = self.lstm(carry, x)
            feature_obs = jnp.concatenate([x, h], axis=-1)
            carry = jnp.concatenate([c, h], axis=-1)  # type: ignore
            return feature_obs, carry

    if params.hidden_state_dim == 0:
        feature_extractor = create_network(
            net=IdentityExtractor(), target_net=IdentityExtractor(), lr=0.0
        )
    else:
        feature_extractor = create_network(
            net=LSTMExtractor(
                observation_space_size, params.hidden_state_dim, nnx.Rngs(params.seed)
            ),
            target_net=LSTMExtractor(
                observation_space_size, params.hidden_state_dim, nnx.Rngs(params.seed)
            ),
            lr=params.feature_extractor_lr,
        )

    actor = create_network(
        net=SimpleNet(
            observation_space_size + params.hidden_state_dim,
            (1 if discrete_action_space else 2) * action_space_size,
            nnx.Rngs(params.seed),
        ),
        target_net=SimpleNet(
            observation_space_size + params.hidden_state_dim,
            (1 if discrete_action_space else 2) * action_space_size,
            nnx.Rngs(params.seed),
        ),
        lr=params.policy_lr,
    )

    if discrete_action_space:
        q1 = create_network(
            net=SimpleNet(
                observation_space_size + params.hidden_state_dim,
                action_space_size,
                nnx.Rngs(params.seed),
            ),
            target_net=SimpleNet(
                observation_space_size + params.hidden_state_dim,
                action_space_size,
                nnx.Rngs(params.seed),
            ),
            lr=params.q_lr,
        )

        q2 = create_network(
            net=SimpleNet(
                observation_space_size + params.hidden_state_dim,
                action_space_size,
                nnx.Rngs(params.seed),
            ),
            target_net=SimpleNet(
                observation_space_size + params.hidden_state_dim,
                action_space_size,
                nnx.Rngs(params.seed),
            ),
            lr=params.q_lr,
        )

        mc_q1 = create_network(
            net=SimpleNet(
                observation_space_size + params.hidden_state_dim,
                action_space_size,
                nnx.Rngs(params.seed),
            ),
            target_net=SimpleNet(
                observation_space_size + params.hidden_state_dim,
                action_space_size,
                nnx.Rngs(params.seed),
            ),
            lr=params.q_lr,
        )

        mc_q2 = create_network(
            net=SimpleNet(
                observation_space_size + params.hidden_state_dim,
                action_space_size,
                nnx.Rngs(params.seed),
            ),
            target_net=SimpleNet(
                observation_space_size + params.hidden_state_dim,
                action_space_size,
                nnx.Rngs(params.seed),
            ),
            lr=params.q_lr,
        )

    else:
        q1 = create_network(
            net=SimpleNet(
                observation_space_size + action_space_size + params.hidden_state_dim,
                1,
                nnx.Rngs(params.seed),
            ),
            target_net=SimpleNet(
                observation_space_size + action_space_size + params.hidden_state_dim,
                1,
                nnx.Rngs(params.seed),
            ),
            lr=params.q_lr,
        )

        q2 = create_network(
            net=SimpleNet(
                observation_space_size + action_space_size + params.hidden_state_dim,
                1,
                nnx.Rngs(params.seed),
            ),
            target_net=SimpleNet(
                observation_space_size + action_space_size + params.hidden_state_dim,
                1,
                nnx.Rngs(params.seed),
            ),
            lr=params.q_lr,
        )

        mc_q1 = create_network(
            net=SimpleNet(
                observation_space_size + action_space_size + params.hidden_state_dim,
                1,
                nnx.Rngs(params.seed),
            ),
            target_net=SimpleNet(
                observation_space_size + action_space_size + params.hidden_state_dim,
                1,
                nnx.Rngs(params.seed),
            ),
            lr=params.q_lr,
        )

        mc_q2 = create_network(
            net=SimpleNet(
                observation_space_size + action_space_size + params.hidden_state_dim,
                1,
                nnx.Rngs(params.seed),
            ),
            target_net=SimpleNet(
                observation_space_size + action_space_size + params.hidden_state_dim,
                1,
                nnx.Rngs(params.seed),
            ),
            lr=params.q_lr,
        )

    buffer, buffer_functions = create_buffer(
        observation_space_size,
        1 if discrete_action_space else action_space_size,
        2 * params.hidden_state_dim,
        params.buffer_size,
        params.batch_size,
    )

    key = jax.random.key(params.seed)
    key, key_reset = jax.random.split(key)
    obs, env_state = env.reset(key_reset, env_params)

    sac_state = SACState(
        feature_extractor=feature_extractor,
        actor=actor,
        q1=q1,
        q2=q2,
        mc_q1=mc_q1,
        mc_q2=mc_q2,
        alpha=create_alpha(params.alpha, params.q_lr),
        buffer=buffer,
        obs=obs,
        hidden_state=jnp.zeros((2 * params.hidden_state_dim,)),
        env_state=env_state,
        random_key=key,
        n_updates=0,
    )
    functions = SACFunctions(learn=learn, predict=predict)

    return sac_state, functions, buffer_functions


@partial(nnx.jit, static_argnums=[2, 3, 4, 5, 7])
def evaluate(
    actor_net,
    feature_net,
    env,
    env_params,
    predict,
    hidden_state_dim,
    key,
    num_episodes,
):
    def evaluate_single_episode(key):
        key_reset, key_episode = jax.random.split(key)
        obs, state = env.reset(key_reset, env_params)

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, hidden_state, state, key = state_input
            key, key_step, key_net = jax.random.split(key, 3)
            action, next_hidden_state = predict(
                obs, hidden_state, actor_net, feature_net, key_net, deterministic=True
            )
            next_obs, next_state, reward, done, _ = env.step(
                key_step, state, action, env_params
            )
            next_hidden_state = jax.lax.select(
                done, jnp.zeros((2 * hidden_state_dim,)), next_hidden_state[0]
            )
            carry = [next_obs, next_hidden_state, next_state, key]
            return carry, [obs, action, reward, next_obs, done]

        # Scan over episode step loop
        _, scan_out = jax.lax.scan(
            policy_step,
            [obs, jnp.zeros((2 * hidden_state_dim)), state, key_episode],
            (),
            501,
        )
        # Return masked sum of rewards accumulated by agent in episode
        obs, action, reward, next_obs, done = scan_out
        first_fail = jnp.argmax(done)
        episode_return = (jnp.where(jnp.arange(501) > first_fail, 0, 1) * reward).sum()
        return episode_return

    def step(carry, tmp):
        key = carry
        key, key_eval = jax.random.split(key)
        episode_len = evaluate_single_episode(key_eval)
        return key, episode_len

    _, episode_lens = jax.lax.scan(step, key, (), num_episodes)
    return episode_lens.mean()
