import gymnax
import jax
import jax.numpy as jnp

import wandb
from lambda_imitation.jax.sac import *

key = jax.random.key(0)
key, key_reset, key_act = jax.random.split(key, 3)

# Instantiate the environment & its settings.
env, env_params = gymnax.make("CartPole-v1")

import time

import numpy as np

import wandb

hyperparameters = Hyperparameters(seed=np.random.randint(1000000))
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="fhstp-data-intelligence-research-group",
    # Set the wandb project where this run will be logged.
    project="jax-sac-cartpole",
    # Track hyperparameters and run metadata.
    config=hyperparameters._asdict(),
)
run.log_code(".")

learn_steps = 1000
sac_state, functions, buffer_functions = create_SAC(
    env, env_params, 4, True, 2, learn_steps, hyperparameters
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
