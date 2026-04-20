import math
from functools import partial
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from buffer import Buffer, BufferSample, create_sample

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class Actor(nnx.Module):
    def __init__(self, obs_dim, action_dim, rngs: nnx.Rngs):
        self.fc_1 = nnx.Linear(obs_dim, 256, rngs=rngs)
        self.fc_2 = nnx.Linear(256, 256, rngs=rngs)
        self.actor_head = nnx.Linear(256, 2 * action_dim, rngs=rngs)

    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.fc_1(x))
        x = nnx.relu(self.fc_2(x))
        dist_params = self.actor_head(x)
        return dist_params


class Critic(nnx.Module):
    def __init__(self, obs_dim, action_dim, rngs: nnx.Rngs):
        self.fc_1 = nnx.Linear(obs_dim, 256, rngs=rngs)
        self.fc_2 = nnx.Linear(256 + action_dim, 256, rngs=rngs)
        self.fc_3 = nnx.Linear(256, 256, rngs=rngs)
        self.critic_head = nnx.Linear(256, 2, rngs=rngs)

    def __call__(self, x, a):
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.fc_1(x))
        x = jnp.concat((x, a), axis=-1)
        x = nnx.relu(self.fc_2(x))
        x = nnx.relu(self.fc_3(x))
        value = self.critic_head(x)
        return value


class IQLearnState(NamedTuple):
    actor: nnx.GraphState
    critic: nnx.GraphState
    actor_target: nnx.GraphState
    critic_target: nnx.GraphState
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    alpha: jax.Array
    log_alpha: jax.Array


class IQLearnFunctions(NamedTuple):
    predict: Callable
    train: Callable


class Hyperparameters(NamedTuple):
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


def create_iqlearn(
    params: Hyperparameters,
    buffer: Buffer,
    action_dim: int,
    obs_key: str = "observations",
    action_key: str = "actions",
    action_scale: float | jax.Array = 1,
    action_bias: float | jax.Array = 0,
    train_steps=1000,
) -> Tuple[IQLearnState, IQLearnFunctions]:
    buffer_sample = create_sample(
        buffer.size,
        params.batch_size,
        this_keys=[obs_key, action_key],
        next_keys=[obs_key],
    )

    obs_dim = math.prod(buffer.info[obs_key].shape[1:])

    actor_model = Actor(obs_dim, action_dim, nnx.Rngs(0))
    critic_model = Critic(obs_dim, action_dim, nnx.Rngs(0))

    actor_graph, actor = nnx.split(actor_model)
    critic_graph, critic = nnx.split(critic_model)

    actor_optimizer = optax.adam(params.actor_lr)
    critic_optimizer = optax.adam(params.critic_lr)
    alpha_optimizer = optax.adam(params.alpha_lr)

    log_alpha = jnp.array(jnp.log(params.alpha))
    actor_optimizer_state = actor_optimizer.init(actor)  # type: ignore
    critic_optimizer_state = critic_optimizer.init(critic)  # type: ignore
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)  # type: ignore

    def remove_weak_types(state):
        state = jax.tree.map(
            lambda x: jnp.array(x, dtype=x.dtype) if hasattr(x, "dtype") else x, state
        )
        return state

    iqlearn = IQLearnState(
        remove_weak_types(actor),
        remove_weak_types(critic),
        remove_weak_types(actor),
        remove_weak_types(critic),
        remove_weak_types(actor_optimizer_state),
        remove_weak_types(critic_optimizer_state),
        remove_weak_types(alpha_optimizer_state),
        remove_weak_types(jnp.exp(log_alpha)),
        remove_weak_types(log_alpha),
    )

    def get_q(
        critic_head: nnx.GraphState,
        x: jax.Array,
        actions: jax.Array,
    ):
        model_critic = nnx.merge(critic_graph, critic_head)
        q = jnp.min(model_critic(x, actions), axis=-1)
        return q

    def get_v(
        actor_head: nnx.GraphState,
        critic_head: nnx.GraphState,
        alpha: jax.Array,
        x: jax.Array,
        key: jax.Array,
        include_entropy: bool = True,
        include_log: bool = False,
    ) -> jax.Array | Tuple[jax.Array, dict]:
        action, logprob = sample_action_logprob(actor_head, x, key)
        q = get_q(critic_head, x, action)
        if include_entropy:
            if include_log:
                # jax.debug.print("act: {x}", x=action[0])
                return q - alpha * logprob, {"q": q.mean(), "entropy": -logprob.mean()}
            else:
                return q - alpha * logprob
        else:
            return q

    def get_dist_params(actor_head: nnx.GraphState, x: jax.Array):
        model_actor = nnx.merge(actor_graph, actor_head)
        dist_params = model_actor(x)
        mean, log_std = (
            dist_params[..., :action_dim],
            dist_params[..., action_dim:],
        )
        log_std = jnp.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = jnp.exp(log_std)
        return mean, std

    def sample_action_logprob(
        actor_head: nnx.GraphState,
        x: jax.Array,
        key: jax.Array,
    ):
        mean, std = get_dist_params(actor_head, x)
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

    @partial(jax.jit, static_argnames=["deterministic"])
    def predict(
        iqlearn: IQLearnState,
        obs: jax.Array,
        key: jax.Array = jnp.array(0),
        deterministic: bool = False,
    ):
        obs = jnp.expand_dims(obs, 0)
        mean, std = get_dist_params(iqlearn.actor, obs)
        if deterministic:
            unsquashed_action = mean
        else:
            unsquashed_action = jax.random.normal(key, mean.shape) * std + mean
        y_t = jnp.tanh(unsquashed_action)
        action = y_t * action_scale + action_bias
        return action[0]

    def loss_alpha(log_alpha, log_pi):
        alpha_loss = -jnp.exp(log_alpha) * (log_pi + params.target_entropy)
        return alpha_loss

    def loss_actor(
        actor_head: nnx.GraphState,
        critic_head: nnx.GraphState,
        buffer: Buffer,
        buffer_sample: Callable[[Buffer, jax.Array], Tuple[BufferSample, Tuple[int]]],
        alpha: jax.Array,
        key: jax.Array,
    ):
        key_sample, key_v = jax.random.split(key, 2)
        sample, _ = buffer_sample(buffer, key_sample)
        v, metrics = get_v(
            actor_head,
            critic_head,
            alpha,
            sample.this_info[obs_key],
            key_v,
            include_entropy=True,
            include_log=True,
        )
        # jax.debug.print("buf: {x}", x=sample.this_info[action_key][0])
        # jax.debug.print("------")
        metrics.update({"v": v.mean()})
        return -v.mean(), metrics

    def loss_critic(
        actor_target: nnx.GraphState,
        critic_head: nnx.GraphState,
        critic_target: nnx.GraphState,
        buffer: Buffer,
        buffer_sample: Callable[[Buffer, jax.Array], Tuple[BufferSample, Tuple[int]]],
        alpha: jax.Array,
        key: jax.Array,
    ):
        key_sample, key_v, key_next_v = jax.random.split(key, 3)
        sample, _ = buffer_sample(buffer, key_sample)

        q_values = get_q(
            critic_head, sample.this_info[obs_key], sample.this_info[action_key]
        )
        v_values = get_v(
            actor_target,
            critic_head,
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
        # next_v_values = jax.lax.stop_gradient(next_v_values)

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

    def update_step(iqlearn: IQLearnState, key):
        print("compiling...")
        # setup
        key_actor, key_critic = jax.random.split(key, 2)

        # grads through update functions
        grads_actor, metrics = jax.grad(loss_actor, has_aux=True)(
            iqlearn.actor,
            iqlearn.critic_target,
            buffer,
            buffer_sample,
            iqlearn.alpha,
            key_actor,
        )
        grads_critic, metrics_critic = jax.grad(loss_critic, argnums=1, has_aux=True)(
            iqlearn.actor_target,
            iqlearn.critic,
            iqlearn.critic_target,
            buffer,
            buffer_sample,
            iqlearn.alpha,
            key_critic,
        )

        # concat metrics
        metrics.update(metrics_critic)

        # update actor
        updates, new_actor_optimizer_state = actor_optimizer.update(
            grads_actor, iqlearn.actor_optimizer_state
        )
        new_actor = optax.apply_updates(iqlearn.actor, updates)  # type: ignore

        # update critic
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
    def train(iqlearn: IQLearnState, key: jax.Array):
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

    return iqlearn, IQLearnFunctions(predict, train), actor_graph


# obs_dim, action_dim = 10, 2
# buffer, buffer_fns = create_buffer(
#     shapes={"observations": (obs_dim,), "actions": (action_dim,)},
#     size=10000, sampling_size=256,
#     this_step_infos=["observations", "actions"],
#     next_step_infos=["observations"],
# )
# iqlearn, functions, _ = create_iqlearn(Hyperparameters(), buffer, action_dim)
# iqlearn, metrics = functions.train(iqlearn, jax.random.key(0))
# print(metrics)
