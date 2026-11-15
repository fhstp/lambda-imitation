"""Whole-episode recurrent learner with explicit, anchor-only training losses.

This is a NEW implementation, not a patch to the sealed historical learner.
All sampler/trace arms share full-prefix reconstruction and exact real terminal
handling. Adaptive replay remains an empirical mixture, not a claimed stationary
mu posterior. Frozen-encoder/actor fits use a separately collected fixed-mu
dataset and do not update the encoder, actor, or task critic.
"""

from dataclasses import dataclass, replace
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from lambda_imitation.actor_critic import (
    AgentState, Head, TwinCriticGraph, TwinCriticState, retrace_targets, trace_coefficients,
)
from lambda_imitation.utils import RecurrentFeatureExtractor, battleship_projection
from lambda_imitation.envs.battleship import Battleship
from lambda_imitation.envs.minesweeper import MineSweeper
from .collection import collect_step, start_rollout
from .replay import create_replay, packed_anchors, select


@dataclass(frozen=True)
class Settings:
    environment: str = "battleship"
    measure: str = "discounted"
    correction: str = "retrace"
    hidden_dim: int = 512
    batch_size: int = 128
    sequence_length: int = 40
    tail: int = 50
    replay_capacity: int = 100_000
    episode_batch_factor: int = 4
    gamma: float = 0.99
    lambda1: float = 0.05
    lambda2: float = 0.85
    lambda_coef: float = 0.01
    alpha: float = 0.1
    tau: float = 0.005
    fe_lr: float = 1e-5
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    frozen: bool = False

    @property
    def actor_count(self): return self.batch_size * (self.sequence_length + self.tail)

    @property
    def task_count(self): return self.batch_size * (self.sequence_length + self.tail - 1)

    @property
    def auxiliary_count(self): return self.batch_size * self.sequence_length


def paper_settings(environment, **changes):
    if environment == "battleship":
        settings = Settings()
    elif environment == "minesweeper":
        settings = Settings(environment=environment, hidden_dim=128, batch_size=64,
                            sequence_length=32, tail=32, lambda1=0.0, lambda2=0.95,
                            lambda_coef=0.1, alpha=0.01)
    else:
        raise ValueError("only paper Battleship and Minesweeper are admitted")
    return replace(settings, **changes)


class Functions(NamedTuple):
    features: object
    probabilities: object
    loss: object
    update: object
    prefill: object
    train: object
    fit: object
    evaluate: object
    sample: object


def create_learner(settings, seed):
    if settings.correction not in ("uncorrected", "is", "retrace"):
        raise ValueError("invalid trace correction")
    if settings.measure not in ("uniform", "discounted"):
        raise ValueError("invalid starting measure")
    if settings.batch_size < 1 or settings.episode_batch_factor < 1:
        raise ValueError("batch dimensions must be positive")
    env = Battleship() if settings.environment == "battleship" else MineSweeper()
    shape = env.observation_space(env.default_params).shape
    actions = env.num_actions
    max_steps = actions if settings.environment == "battleship" else env.episode_length
    obs_fn = (lambda x: x[..., :1]) if settings.environment == "battleship" else lambda x: x
    mask_fn = (lambda x: x[..., 1:].astype(bool)) if settings.environment == "battleship" else None
    width = settings.hidden_dim
    projection = battleship_projection(width) if settings.environment == "battleship" else width
    key_fe, key_heads = jax.random.split(jax.random.key(seed))
    fe = RecurrentFeatureExtractor((1,) if mask_fn else shape, projection, "gru", width,
                                   actions, rngs=nnx.Rngs(key_fe))
    fe_graph, fe_state = nnx.split(fe)
    keys = jax.random.split(key_heads, 7)

    def head(key, norm=False):
        return nnx.split(Head(width, (width,), actions, layer_norm=norm, rngs=nnx.Rngs(key)))

    def twin(k1, k2):
        g1, s1 = head(k1, True)
        g2, s2 = head(k2, True)
        return TwinCriticGraph(g1, g2), TwinCriticState(s1, s2)

    actor_graph, actor_state = head(keys[0])
    graphs, states = [], []
    for i in (1, 3, 5):
        graph, state = twin(keys[i], keys[i+1])
        graphs.append(graph)
        states.append(state)
    networks = (fe_state, actor_state, *states)
    optimizers = [optax.adam(lr) for lr in
                  (settings.fe_lr, settings.actor_lr, settings.critic_lr,
                   settings.critic_lr, settings.critic_lr)]
    opts = tuple(opt.init(state) for opt, state in zip(optimizers, networks))
    agent = AgentState(*networks, *networks, *opts, create_replay(shape, max_steps, settings.replay_capacity), jnp.int32(0))

    def policy(actor, features, masks=None):
        logits = nnx.merge(actor_graph, actor)(features)
        if masks is not None:
            # Only zero-padding may be all-masked. Actual collection is checked
            # independently against the environment's real pre-action support.
            masks = masks | ~masks.any(-1, keepdims=True)
            logits = jnp.where(masks, logits, -1e9)
        return jax.nn.softmax(logits), jax.nn.log_softmax(logits)

    def qs(params, graph, features):
        return jnp.stack([nnx.merge(graph.q1, params.q1)(features),
                          nnx.merge(graph.q2, params.q2)(features)], axis=-1)

    def chosen(params, graph, features, taken):
        values = qs(params, graph, features)
        return values[jnp.arange(taken.shape[0]), taken.astype(jnp.int32)]

    def reconstruct(fe_state, info, lengths):
        model = nnx.merge(fe_graph, fe_state)
        observations = jnp.swapaxes(obs_fn(info["observations"]), 0, 1)
        encoded = jax.nn.one_hot(info["actions"], actions)
        previous = jnp.concatenate([jnp.zeros_like(encoded[:, :1]), encoded[:, :-1]], axis=1)
        previous = jnp.swapaxes(previous, 0, 1)
        valid = jnp.arange(max_steps)[:, None] < lengths[None, :]

        def step(carry, inputs):
            obs, pa, active = inputs
            next_carry, features = model(carry, obs, pa)
            return jnp.where(active[:, None], next_carry, carry), jnp.where(active[:, None], features, 0)

        _, result = jax.lax.scan(step, jnp.zeros((lengths.shape[0], width)),
                                 (observations, previous, valid))
        return jnp.swapaxes(result, 0, 1)

    def probabilities(agent, observation, pending, age):
        data = {**pending, "observations": pending["observations"].at[age].set(observation)}
        data = {name: value[None] for name, value in data.items()}
        features = reconstruct(agent.feature_extractor, data, jnp.atleast_1d(age + 1))[0, age]
        masks = mask_fn(observation)[None] if mask_fn else None
        return policy(agent.actor, features[None], masks)[0][0]

    count = settings.auxiliary_count if settings.frozen else settings.actor_count

    def sample(replay, key):
        return packed_anchors(replay, key, gamma=settings.gamma, measure=settings.measure,
                              episode_batch=settings.batch_size * settings.episode_batch_factor,
                              score_count=count)

    def loss(fe, actor, critic, l1, l2, agent, key, batch=None):
        batch = sample(agent.online_buffer, key) if batch is None else batch
        info = batch.info
        latent = reconstruct(fe, info, batch.lengths)
        target_latent = reconstruct(agent.feature_extractor_target, info, batch.lengths)
        if settings.frozen:
            latent = jax.lax.stop_gradient(latent)
        masks = mask_fn(info["observations"]) if mask_fn else None
        flat = lambda x: x.reshape((-1, *x.shape[2:]))
        target_probs, _ = policy(agent.actor_target, flat(target_latent), flat(masks) if mask_fn else None)
        target_probs = target_probs.reshape((*batch.valid.shape, actions))
        ratios = jnp.take_along_axis(target_probs, info["actions"][..., None], -1)[..., 0] / info["behaviour_probs"]
        targets = []
        for params, graph in zip((agent.critic_target, agent.lambda1_critic_target,
                                  agent.lambda2_critic_target), graphs):
            values = qs(params, graph, flat(target_latent)).min(-1).reshape(target_probs.shape)
            taken = jnp.take_along_axis(values, info["actions"][..., None], -1)[..., 0]
            expected = (values * target_probs).sum(-1)
            targets.append((taken, expected))

        result, metrics = jnp.float32(0), {}
        aux_z = select(latent, batch, settings.auxiliary_count)
        aux_actions = select(info["actions"], batch, settings.auxiliary_count)
        for i, (params, lam) in enumerate(((l1, settings.lambda1), (l2, settings.lambda2)), start=1):
            returns = retrace_targets(targets[i][0].T, targets[i][1].T,
                                      info["rewards"].T, info["terminated"].T, ratios.T,
                                      settings.gamma, lam, correction=settings.correction).T
            expected = jax.lax.stop_gradient(select(returns, batch, settings.auxiliary_count))
            predictions = chosen(params, graphs[i], aux_z, aux_actions)
            regression = 0.5 * ((predictions - expected[:, None]) ** 2).sum(-1).mean()
            result += regression
            metrics[f"lambda{i}_loss"] = regression
            metrics[f"lambda{i}_target_abs_max"] = jnp.abs(expected).max()

        if not settings.frozen:
            actor_z = jax.lax.stop_gradient(select(latent, batch, settings.actor_count))
            actor_masks = select(masks, batch, settings.actor_count) if mask_fn else None
            probs, log_probs = policy(actor, actor_z, actor_masks)
            actor_q = qs(jax.lax.stop_gradient(critic), graphs[0], actor_z).min(-1)
            entropy = -(probs * log_probs).sum(-1)
            actor_loss = -((probs * actor_q).sum(-1) + settings.alpha * entropy).mean()
            task_z = select(latent, batch, settings.task_count)
            task_actions = select(info["actions"], batch, settings.task_count)
            next_v = jnp.concatenate([targets[0][1][:, 1:], jnp.zeros_like(targets[0][1][:, :1])], axis=1)
            td_target = info["rewards"] + settings.gamma * (1 - info["terminated"]) * next_v
            td_target = jax.lax.stop_gradient(select(td_target, batch, settings.task_count))
            task_predictions = chosen(critic, graphs[0], task_z, task_actions)
            task_loss = 0.5 * ((task_predictions - td_target[:, None]) ** 2).sum(-1).mean()
            d = (chosen(jax.lax.stop_gradient(l1), graphs[1], task_z, task_actions).min(-1)
                 - chosen(jax.lax.stop_gradient(l2), graphs[2], task_z, task_actions).min(-1))
            discrepancy = (d ** 2).mean()
            result += actor_loss + task_loss + settings.lambda_coef * discrepancy
            metrics.update(actor_loss=actor_loss, task_loss=task_loss,
                           discrepancy_loss=discrepancy, entropy=entropy.mean())

        selected_ratios = select(ratios, batch)
        scaled = selected_ratios / jnp.maximum(selected_ratios.max(), 1e-30)
        metrics.update(importance_ratio_max=selected_ratios.max(),
                       importance_ratio_mean=selected_ratios.mean(),
                       importance_ratio_ess_fraction=scaled.sum() ** 2 / jnp.maximum((scaled**2).sum(), 1e-30) / scaled.size,
                       anchor_age_mean=batch.times.astype(jnp.float32).mean(),
                       retained_transitions=agent.online_buffer.transitions.astype(jnp.float32))
        ages = jnp.arange(max_steps)
        observed = (batch.times[:, None] == ages[None, :]).mean(0)
        length_counts = jnp.bincount(agent.online_buffer.lengths, length=max_steps+1)
        eligible_at_age = jnp.cumsum(length_counts[::-1])[::-1][1:]
        expected = eligible_at_age * (settings.gamma**ages if settings.measure == "discounted" else 1)
        expected = expected / expected.sum()
        metrics["expected_anchor_age_mean"] = (expected*ages).sum()
        metrics["trace_coefficient_max"] = jnp.where(batch.valid,
            trace_coefficients(ratios, settings.lambda2, settings.correction), 0).max()
        for age in range(max_steps):
            metrics[f"anchor_age_{age:02d}_fraction"] = observed[age]
            metrics[f"expected_anchor_age_{age:02d}_fraction"] = expected[age]
        return result, metrics

    @jax.jit
    def update(agent, key):
        (value, metrics), grads = jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4), has_aux=True)(
            *agent[:5], agent, key)
        networks, opts = [], []
        for i, (opt, grad, params, state) in enumerate(zip(optimizers, grads, agent[:5], agent[10:15])):
            if settings.frozen and i < 3:
                networks.append(params)
                opts.append(state)
            else:
                changes, new_opt = opt.update(grad, state, params)
                networks.append(optax.apply_updates(params, changes))
                opts.append(new_opt)
        target_networks = tuple(jax.tree.map(lambda old, new: old if settings.frozen and i < 3 else
            (1-settings.tau)*old + settings.tau*new, old, new)
            for i, (old, new) in enumerate(zip(agent[5:10], networks)))
        metrics = {**metrics, "total_loss": value}
        return AgentState(*networks, *target_networks, *opts, agent.online_buffer, agent.update_step + 1), metrics

    @partial(jax.jit, static_argnames=["steps"])
    def prefill(agent, rollout, *, steps):
        def step(state, _):
            a, r = state
            replay, r = collect_step(a.online_buffer, r, env, None, action_dim=actions,
                                     mask_fn=mask_fn, uniform=True)
            return (a._replace(online_buffer=replay), r), None
        return jax.lax.scan(step, (agent, rollout), None, length=steps)[0]

    def reduce(metrics):
        return {name: values.max() if name.endswith("_max") else values.mean()
                for name, values in metrics.items()}

    @partial(jax.jit, static_argnames=["steps"])
    def train(agent, rollout, key, *, steps):
        if settings.frozen:
            raise ValueError("frozen fitting cannot collect adaptive data")
        def step(state, _):
            a, r, k = state
            k, update_key = jax.random.split(k)
            replay, r = collect_step(a.online_buffer, r, env, lambda *x: probabilities(a, *x),
                action_dim=actions, mask_fn=mask_fn, collector_version=a.update_step)
            a, metrics = update(a._replace(online_buffer=replay), update_key)
            return (a, r, k), metrics
        (agent, rollout, key), metrics = jax.lax.scan(step, (agent, rollout, key), None, length=steps)
        return agent, rollout, key, reduce(metrics)

    @partial(jax.jit, static_argnames=["steps"])
    def fit(agent, key, *, steps):
        if not settings.frozen:
            raise ValueError("critic-only fitting requires an explicitly frozen configuration")
        def step(state, _):
            a, k = state
            k, sample_key = jax.random.split(k)
            a, metrics = update(a, sample_key)
            return (a, k), metrics
        (agent, key), metrics = jax.lax.scan(step, (agent, key), None, length=steps)
        return agent, key, reduce(metrics)

    @partial(jax.jit, static_argnames=["episodes", "greedy"])
    def evaluate(agent, key, *, episodes, greedy):
        model = nnx.merge(fe_graph, agent.feature_extractor)
        def episode(key):
            key, rk = jax.random.split(key)
            observation, es = env.reset(rk, env.default_params)
            def step(state, age):
                obs, es, carry, pa, key, active, total = state
                key, ak, ek = jax.random.split(key, 3)
                carry, z = model(carry, obs[None, :1] if mask_fn else obs[None], pa[None])
                masks = mask_fn(obs)[None] if mask_fn else None
                probs, _ = policy(agent.actor, z, masks)
                action = jnp.argmax(probs[0]) if greedy else jax.random.categorical(ak, jnp.log(probs[0]))
                gap = (qs(agent.lambda1_critic, graphs[1], z).min(-1)
                       - qs(agent.lambda2_critic, graphs[2], z).min(-1))[0]
                d2 = (probs[0] * gap**2).sum()
                obs, es, reward, done, _ = env.step(ek, es, action, env.default_params)
                total += active * jnp.array([reward, 1.0, done, d2,
                                             settings.gamma**age*d2, settings.gamma**age])
                return (obs, es, carry, jax.nn.one_hot(action, actions), key, active & ~done, total), None
            initial = (observation, es, jnp.zeros((1, width)), jnp.zeros(actions), key,
                       jnp.bool_(True), jnp.zeros(6))
            final, _ = jax.lax.scan(step, initial, jnp.arange(max_steps))
            return final[-1]
        result = jax.vmap(episode)(jax.random.split(key, episodes))
        total = result.sum(0)
        return dict(return_=result[:, 0].mean(), steps=result[:, 1].mean(),
                    completed=result[:, 2].mean(),
                    discrepancy_rms=jnp.sqrt(total[3]/total[1]),
                    discounted_discrepancy_rms=jnp.sqrt(total[4]/total[5]))

    rollout = start_rollout(env, jax.random.fold_in(jax.random.key(seed), 0xC011), shape, max_steps)
    return agent, rollout, Functions(reconstruct, probabilities, loss, update, prefill,
                                     train, fit, evaluate, sample)
