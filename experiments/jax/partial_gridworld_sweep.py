import sys
import time

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import yaml

import wandb
from lambda_imitation.jax.gridworld import PartiallyObservableGridworld
from lambda_imitation.jax.sac import *


def train(config):
    key = jax.random.key(0)
    key, key_reset, key_act = jax.random.split(key, 3)

    # Instantiate the environment & its settings.
    env = PartiallyObservableGridworld()
    env_params = env.default_params
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="fhstp-data-intelligence-research-group",
        # Set the wandb project where this run will be logged.
        project="jax-sac-partial-gridworld-sweep",
    )
    args = run.config
    hyperparameters = Hyperparameters(**args)
    run.log_code(".")

    learn_steps = 1000
    sac_state, functions, buffer_functions = create_SAC(
        env, env_params, 9, True, 4, learn_steps, hyperparameters
    )

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
    run.log(log, step=0)

    # Perform the step transition.
    buffer = sac_state.buffer
    env_state = sac_state.env_state
    obs = sac_state.obs
    hidden_state = sac_state.hidden_state
    for _ in range(hyperparameters.learning_starts):
        key_act, split, key_env = jax.random.split(key_act, 3)
        action = env.action_space(env_params).sample(split)  # type: ignore
        buffer, obs, done, env_state = run_env_step(
            env,
            env_params,
            action,
            0.5,
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

    sac_state = SACState(
        feature_extractor=sac_state.feature_extractor,
        actor=sac_state.actor,
        q1=sac_state.q1,
        q2=sac_state.q2,
        mc_q1=sac_state.mc_q1,
        mc_q2=sac_state.mc_q2,
        alpha=sac_state.alpha,
        buffer=buffer,
        obs=sac_state.obs,
        hidden_state=sac_state.hidden_state,
        env_state=sac_state.env_state,
        random_key=sac_state.random_key,
        n_updates=sac_state.n_updates,
    )

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
        # print(metrics["train/mc_q_values"])
        run.log(log, step=(i + 1) * learn_steps)


wandb.agent("o0npuxy3", train, project="jax-sac-partial-gridworld-sweep")
