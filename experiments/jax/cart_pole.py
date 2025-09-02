import gymnax
import jax
import jax.numpy as jnp

import wandb
from lambda_imitation.jax.sac import *

key = jax.random.key(0)
key, key_reset, key_act = jax.random.split(key, 3)

# Instantiate the environment & its settings.
env, env_params = gymnax.make("CartPole-v1")

import sys
import time

import numpy as np
import yaml

import wandb

args = {}
if len(sys.argv) > 1:
    with open(sys.argv[1]) as file:
        args = yaml.safe_load(file.read())
args["seed"] = np.random.randint(1000000)

hyperparameters = Hyperparameters(**args)
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="fhstp-data-intelligence-research-group",
    # Set the wandb project where this run will be logged.
    project="jax-sac-partial-cartpole",
    # Track hyperparameters and run metadata.
    config=hyperparameters._asdict(),
)
run.log_code(".")

learn_steps = 1000
sac_state, functions, buffer_functions = create_SAC(
    env, env_params, 4, True, 2, learn_steps, hyperparameters
)


class LSTMExtractor(nnx.Module):
    def __init__(self, din, dout, rngs):
        self.lstm = nnx.OptimizedLSTMCell(din, dout, rngs=rngs)

    def __call__(self, carry, x):
        x = x.at[..., 3].set(0).at[..., 1].set(0)
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
        4, hyperparameters.hidden_state_dim, nnx.Rngs(hyperparameters.seed)
    ),
    target_net=LSTMExtractor(
        4, hyperparameters.hidden_state_dim, nnx.Rngs(hyperparameters.seed)
    ),
    lr=hyperparameters.feature_extractor_lr,
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
        hyperparameters.gamma,
        hyperparameters.lambda_discrepancy_coef > 0.0,
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
