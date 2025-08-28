from functools import partial
from typing import Any, Callable, NamedTuple, Tuple

import flax.nnx as nnx
import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from lambda_imitation.jax.buffer import (
    Buffer,
    BufferFunctions,
    BufferSample,
    create_buffer,
)


class Hyperparameters(NamedTuple):
    seed: int = 42
    """The seed used"""
    buffer_size: int = 100000
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
    alpha: float = 1.0
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy: float = 0.2
    """The target entropy when not chosen automatically"""
    hidden_state_dim: int = 0
    """Dimension of hidden states, has to be even, if 0, no LSTM will be used"""
    hidden_state_recalculation_interval: int = 500
    """How often the hidden states of the demonstration buffer are recalculated"""
    recalculate_hidden_states_in_update: bool = False
    """Whether or not to recalculate hidden states at every step for sample"""
    use_lambda_discrepancy: bool = False
    """Whether or not to also approximate the value function via MC estimation and use lambda discrepancy to optimize memory"""
    use_action_recalculation: bool = False
    """Whether or not to re-calculate actions in calculation of lambda-discrepancy"""
    update_feature_extractor_with_losses: bool = False
    """Whether or not the feature extractor gets updated with standard losses or just by lambda-discrepancy. If `use_lambda_discrepancy` is False, this will always be True"""
    use_importance_sampling: bool = True
    """Whether or not to use importance sampling for MC approximation"""


class Network(NamedTuple):
    net: nnx.Module
    target_net: nnx.Module | None
    optimizer: nnx.Optimizer


class Alpha(NamedTuple):
    log_alpha: jnp.float32
    alpha: jnp.float32
    optimizer_update: Any
    optimizer_state: Any


def create_alpha(start_alpha, lr):
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
    alpha: Alpha
    buffer: Buffer
    obs: jax.Array
    env_state: Any
    random_key: Any
    n_updates: int


class SACFunctions(NamedTuple):
    learn: Callable
    predict: Callable


@partial(nnx.jit, static_argnums=[0, 1, 6])
def run_env_step(
    env,
    env_params,
    action,
    buffer: Buffer,
    current_obs,
    env_state,
    buffer_functions,
    key,
) -> Tuple[Buffer, jax.Array, jax.Array]:
    key_step, key_reset = jax.random.split(key)
    obs, state, reward, done, _ = env.step(key_step, env_state, action, env_params)
    new_buffer: Buffer = buffer_functions.add(buffer, current_obs, action, reward, done)
    reset_obs, reset_state = env.reset(key_reset, env_params)
    return_obs = jax.tree.map(lambda x, y: done * x + (1 - done) * y, reset_obs, obs)
    return_state = jax.tree.map(
        lambda x, y: done * x + (1 - done) * y, reset_state, state
    )
    return new_buffer, return_obs, return_state


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
            mean, log_std = actor_net(observations)
            std = log_std.exp()
            x_t = jax.random.normal(key, mean.shape) * std + mean
            y_t = jnp.tanh(x_t)
            action = y_t * action_scale + action_bias
            log_prob = (
                -(x_t**2) / 2
                + jnp.sqrt(2 * jnp.pi)
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
        q1_target_net: nnx.Module,
        q2_target_net: nnx.Module,
        actor_net: nnx.Module,
        alpha: float,
        buffer: Buffer,
        key,
    ):
        key_sample, key_next_actions = jax.random.split(key)
        sample: BufferSample = buffer_functions.sample(buffer, key_sample)
        if discrete_action_space:
            logits = actor_net(sample.next_observations)  # type: ignore
            probs = jax.nn.softmax(logits, axis=1)
            log_prob = jax.nn.log_softmax(logits, axis=1)
            qf1_next_target = (q1_target_net if params.use_q_targets else q1_net)(  # type: ignore
                sample.next_observations
            )
            qf2_next_target = (q2_target_net if params.use_q_targets else q2_net)(  # type: ignore
                sample.next_observations
            )
            min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target)
            target_q_values = sample.rewards + (
                1 - sample.terminated
            ) * params.gamma * ((min_qf_next_target - alpha * log_prob) * probs).sum(-1)
            target_q_values = jax.lax.stop_gradient(target_q_values)
        else:
            mean, log_std = actor_net(sample.next_observations)  # type: ignore
            std = log_std.exp()
            x_t = jax.random.normal(key_next_actions, mean.shape) * std + mean
            y_t = jnp.tanh(x_t)
            next_actions = y_t * action_scale + action_bias
            log_prob = (
                -(x_t**2) / 2
                + jnp.sqrt(2 * jnp.pi)
                - jnp.log(action_scale * (1 - y_t**2) + 1e-6)
            )
            qf1_next_target = get_q_values(
                sample.next_observations,
                next_actions,
                q1_target_net if params.use_q_targets else q1_net,
            )
            qf2_next_target = get_q_values(
                sample.next_observations,
                next_actions,
                q2_target_net if params.use_q_targets else q2_net,
            )
            min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target)
            target_q_values = sample.rewards + (
                1 - sample.terminated
            ) * params.gamma * (min_qf_next_target - alpha * log_prob)
            target_q_values = jax.lax.stop_gradient(target_q_values)

        qf1_values = get_q_values(sample.observations, sample.actions, q1_net)
        qf2_values = get_q_values(sample.observations, sample.actions, q2_net)

        q1_loss = (qf1_values - target_q_values) ** 2
        q2_loss = (qf2_values - target_q_values) ** 2
        return (q1_loss + q2_loss).mean(), (qf1_values, qf2_values)

    def loss_alpha(
        log_alpha: jax.Array,
        actor_net: nnx.Module,
        buffer: Buffer,
        key,
    ):
        sample: BufferSample = buffer_functions.sample(buffer, key)
        if discrete_action_space:
            logits = actor_net(sample.observations)
            action_probs = jax.nn.softmax(logits, axis=1)
            log_prob = jax.nn.log_softmax(logits, axis=1)
            entropy = (log_prob * action_probs).sum(axis=1).mean()
            alpha_loss = -jnp.exp(log_alpha) * (entropy + params.target_entropy)
            return alpha_loss, entropy
        else:
            _, log_pi = get_action(sample.observations, actor_net, key)
            alpha_loss = -jnp.exp(log_alpha) * (log_pi.mean() + params.target_entropy)
            return alpha_loss, log_pi.mean()

    def loss_actor(
        actor_net: nnx.Module,
        q1_net: nnx.Module,
        q2_net: nnx.Module,
        alpha: float,
        buffer: Buffer,
        key,
    ):
        sample: BufferSample = buffer_functions.sample(buffer, key)
        if discrete_action_space:
            qf1_pi = q1_net(sample.observations)
            qf2_pi = q2_net(sample.observations)
            min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)

            logits = actor_net(sample.observations)
            action_probs = jax.nn.softmax(logits, axis=1)
            log_prob = jax.nn.log_softmax(logits, axis=1)
            # jax.debug.print("{x}", x=action_probs[0])
            return (((alpha * log_prob) - min_qf_pi) * action_probs).sum(axis=-1).mean()
        else:
            pi, log_pi = get_action(sample.observations, actor_net, key)
            xa = jnp.concatenate([sample.observations, pi], axis=-1)
            qf1_pi = q1_net(xa)
            qf2_pi = q2_net(xa)
            min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)
            return ((alpha * log_pi) - min_qf_pi).mean()

    @nnx.jit
    def learn(
        state: SACState,
    ):
        def update_step(state: SACState, tmp):
            print(state.random_key)
            key_next, key_q, key_actor, key_action, key_env, key_alpha = (
                jax.random.split(state.random_key, 6)
            )
            action = predict(state.obs, None, state.actor.net, key_action)
            buffer, obs, env_state = run_env_step(
                env,
                env_params,
                action,
                state.buffer,
                state.obs,
                state.env_state,
                buffer_functions,
                key_env,
            )

            (value_loss_q, (q1_values, q2_values)), grads_q = nnx.value_and_grad(
                loss_q, has_aux=True, argnums=[0, 1]
            )(
                state.q1.net,
                state.q2.net,
                state.q1.target_net,
                state.q2.target_net,
                state.actor.net,
                state.alpha.alpha,
                buffer,
                key_q,
            )
            state.q1.optimizer.update(state.q1.net, grads_q[0])
            state.q2.optimizer.update(state.q2.net, grads_q[1])

            value_loss_actor, grads_actor = nnx.value_and_grad(loss_actor)(
                state.actor.net,
                state.q1.net,
                state.q2.net,
                state.alpha.alpha,
                buffer,
                key_actor,
            )
            state.actor.optimizer.update(state.actor.net, grads_actor)

            if params.autotune:
                (value_loss_alpha, entropy), grads_alpha = nnx.value_and_grad(
                    loss_alpha, has_aux=True
                )(
                    state.alpha.log_alpha,
                    state.actor.net,
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

            return (
                SACState(
                    feature_extractor=state.feature_extractor,
                    actor=state.actor,
                    q1=new_q1,
                    q2=new_q2,
                    alpha=new_alpha,
                    buffer=buffer,
                    obs=obs,
                    env_state=env_state,
                    random_key=key_next,
                    n_updates=state.n_updates + 1,
                ),
                {
                    "loss_q": value_loss_q,
                    "q1_values": q1_values.mean(),
                    "q2_values": q2_values.mean(),
                    "loss_actor": value_loss_actor,
                    "loss_alpha": value_loss_alpha,
                    "entropy": -entropy,
                },
            )

        state, metrics = jax.lax.scan(update_step, state, None, learn_steps)
        return state, metrics

    def predict(obs, hidden_state, actor_net, key, deterministic=False) -> jax.Array:
        obs = obs.reshape(1, -1)
        if discrete_action_space:
            logits = actor_net(obs)
            if deterministic:
                return jnp.argmax(logits)
            else:
                return jax.random.categorical(key, logits)[0]
        else:
            mean, log_std = actor_net(obs)
            if deterministic:
                return mean.reshape(-1)
            std = log_std.exp()
            x_t = jax.random.normal(key, mean.shape) * std + mean
            y_t = jnp.tanh(x_t)
            action = y_t * action_scale + action_bias
            return action.reshape(-1)
        pass

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

        def __call__(self, x):
            return x

    feature_extractor = create_network(
        net=IdentityExtractor(), target_net=IdentityExtractor(), lr=0.0
    )

    actor = create_network(
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
        lr=params.policy_lr,
    )

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

    buffer, buffer_functions = create_buffer(
        observation_space_size,
        1 if discrete_action_space else action_space_size,
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
        alpha=create_alpha(params.alpha, params.q_lr),
        buffer=buffer,
        obs=obs,
        env_state=env_state,
        random_key=key,
        n_updates=0,
    )
    functions = SACFunctions(learn=learn, predict=predict)

    return sac_state, functions, buffer_functions


@partial(nnx.jit, static_argnums=[1, 2, 3, 5])
def evaluate(actor_net, env, env_params, predict, key, num_episodes):
    def evaluate_single_episode(key):
        key_reset, key_episode = jax.random.split(key)
        obs, state = env.reset(key_reset, env_params)

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_params, key = state_input
            key, key_step, key_net = jax.random.split(key, 3)
            action = predict(obs, None, actor_net, key_net, deterministic=True)
            next_obs, next_state, reward, done, _ = env.step(
                key_step, state, action, env_params
            )
            carry = [next_obs, next_state, policy_params, key]
            return carry, [obs, action, reward, next_obs, done]

        # Scan over episode step loop
        _, scan_out = jax.lax.scan(
            policy_step, [obs, state, sac_state, key_episode], (), 501
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


key = jax.random.key(0)
key, key_reset, key_act = jax.random.split(key, 3)

# Instantiate the environment & its settings.
env, env_params = gymnax.make("CartPole-v1")

sac_state, functions, buffer_functions = create_SAC(
    env, env_params, 4, True, 2, 10000, Hyperparameters()
)

print(evaluate(sac_state.actor.net, env, env_params, functions.predict, key, 5))

# Perform the step transition.
buffer = sac_state.buffer
env_state = sac_state.env_state
obs = sac_state.obs
for _ in range(256):
    key_act, split, key_env = jax.random.split(key_act, 3)
    action = env.action_space(env_params).sample(split)
    buffer, obs, env_state = run_env_step(
        env,
        env_params,
        action,
        buffer,
        obs,
        env_state,
        buffer_functions,
        key_env,
    )

for i in range(100):
    sac_state, metrics = functions.learn(sac_state)
    # plt.plot(metrics["q1_values"])
    # plt.show()
    # print(metrics)
    print(
        f"{i}: {evaluate(sac_state.actor.net, env, env_params, functions.predict, key, 5)}"
    )
