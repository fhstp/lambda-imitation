import gymnax
import jax
import jax.numpy as jnp

import wandb
from lambda_imitation.jax.sac import *

key = jax.random.key(0)
key, key_reset, key_act = jax.random.split(key, 3)

# Instantiate the environment & its settings.
env, env_params = gymnax.make("Pendulum-v1")

import time

import numpy as np

import wandb

hyperparameters = Hyperparameters(
    seed=np.random.randint(1000000),
    target_entropy=-1.0,
    hidden_state_dim=10,
    policy_update_frequency=10,
)
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="fhstp-data-intelligence-research-group",
    # Set the wandb project where this run will be logged.
    project="jax-sac-pendulum",
    # Track hyperparameters and run metadata.
    config=hyperparameters._asdict(),
)
run.log_code(".")

learn_steps = 10000
sac_state, functions, buffer_functions = create_SAC(
    env, env_params, 3, False, 1, learn_steps, hyperparameters, 4.0, 2.0
)


class LSTMExtractor(nnx.Module):
    def __init__(self, din, dout, rngs):
        self.lstm = nnx.OptimizedLSTMCell(din, dout, rngs=rngs)

    def __call__(self, carry, x):
        x = x.at[..., -1].set(0)
        carry = (
            carry[..., : hyperparameters.hidden_state_dim],
            carry[..., hyperparameters.hidden_state_dim :],
        )
        (c, h), _ = self.lstm(carry, x)
        feature_obs = jnp.concatenate([x, h], axis=-1)
        carry = jnp.concatenate([c, h], axis=-1)  # type: ignore
        return feature_obs, carry


feature_extractor = create_network(
    net=LSTMExtractor(
        3, hyperparameters.hidden_state_dim, nnx.Rngs(hyperparameters.seed)
    ),
    target_net=LSTMExtractor(
        3, hyperparameters.hidden_state_dim, nnx.Rngs(hyperparameters.seed)
    ),
    lr=hyperparameters.feature_extractor_lr,
)

sac_state = SACState(
    feature_extractor=feature_extractor,
    actor=sac_state.actor,
    q1=sac_state.q1,
    q2=sac_state.q2,
    alpha=sac_state.alpha,
    buffer=sac_state.buffer,
    obs=sac_state.obs,
    hidden_state=sac_state.hidden_state,
    env_state=sac_state.env_state,
    random_key=sac_state.random_key,
    n_updates=sac_state.n_updates,
)
print(
    evaluate(
        sac_state.actor.net,
        sac_state.feature_extractor.net,
        env,
        env_params,
        functions.predict,
        hyperparameters.hidden_state_dim,
        key,
        5,
    )
)


# Perform the step transition.
buffer = sac_state.buffer
env_state = sac_state.env_state
obs = sac_state.obs
hidden_state = sac_state.hidden_state
for _ in range(256):
    key_act, split, key_env = jax.random.split(key_act, 3)
    action = env.action_space(env_params).sample(split)  # type: ignore
    buffer, obs, done, env_state = run_env_step(
        env,
        env_params,
        action,
        buffer,
        obs,
        hidden_state,
        env_state,
        buffer_functions,
        key_env,
    )
    _, hidden_state = sac_state.feature_extractor.net(hidden_state, obs)  # type: ignore

from tqdm.rich import tqdm

for i in tqdm(range(1000)):
    sac_state, metrics = functions.learn(sac_state)
    # plt.plot(metrics["q1_values"])
    # plt.show()
    # print(metrics)
    split, key = jax.random.split(split)
    eval = evaluate(
        sac_state.actor.net,
        sac_state.feature_extractor.net,
        env,
        env_params,
        functions.predict,
        hyperparameters.hidden_state_dim,
        key,
        5,
    )
    log = {"eval/return": eval}
    for key in metrics:
        log[key] = metrics[key].mean()
    run.log(log, step=i * learn_steps)
