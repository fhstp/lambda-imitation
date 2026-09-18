"""Selected 5x5 AC+lambda computation, extracted from the F5-lambda-off run.

Pure JAX state transitions; no study controllers, alternative objectives or probes.
See docs/ac-lambda.md for the exact source map and numerical validation scope.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
from typing import Any, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from . import episode_replay as replay
from .iqlearn import Head, TwinCriticState
from .utils import RecurrentFeatureExtractor, battleship_projection

stop = jax.lax.stop_gradient
OBJECTIVES = ("actor", "standard_td", "lambda1_td", "lambda2_td", "discrepancy")
GROUPS = ("actor_fe", "actor", "standard", "lambda1", "lambda2")


@dataclass(frozen=True)
class Settings:
    width: int = 512
    actions: int = 25
    board_rows: int = 5
    board_cols: int = 5
    ships: tuple = (3, 2)
    gamma: float = .99
    alpha: float = .1
    lambdas: tuple = (.1, .95)
    c_bar: float = 1.1
    rho_bar: float = 1.1  # Diagnostics only; corrected lambda G has no current rho.
    fe_learning_rate: float = 1e-5
    head_learning_rate: float = 1e-4
    b1: float = .9
    b2: float = .999
    eps: float = 1e-8
    eps_root: float = 0.
    tau: float = .001
    huber_delta: float = 1.
    replay_capacity: int = 200000
    batch_episodes: int = 128
    sequence_length: int = 25
    prefill: int = 16640
    reporting_interval: int = 10000
    evaluation_episodes: int = 500
    terminal_reward: float = 25.
    nonterminal_reward: float = -1.
    critic_layer_norm: bool = True
    layer_norm_epsilon: float = 1e-6
    fake_onpolicy_loss: bool = False
    entropy_bootstrap: bool = False
    gradient_clipping: Any = None
    annealing: bool = False
    burn_in_length: int = 0
    target_tail: int = 0
    objective_weights: tuple = (1., 1., 1., 1., 1.)


def resolved_config(seed=0, total=500000):
    if type(seed) is not int or seed not in range(10) or total not in (500000, 2000000):
        raise ValueError("seed must index split(PRNGKey(2029),10); total must be 500000 or 2000000")
    return dict(seed=seed, total=total, settings=asdict(Settings()),
                root_seed=2029, collection_domain=1, evaluation_domain=2, evaluation_folds=[0xE24, 0xA24],
                checkpoints=[n for n in (0,100000,250000,500000,750000,1000000,1500000,2000000) if n <= total],
                architecture="hit+previous-action -> Linear1024/ReLU -> hit-skip/Linear512/ReLU -> GRU512",
                critics="three independent twins; hidden Linear512/LayerNorm/ReLU; min aggregation",
                actor="masked expected Q; entire Q teacher stopped; actor/entropy FE gradients active",
                lambda_target="Q-conditioned G with next-action c=lambda*min(1.1,pi_EMA/mu), no current rho",
                losses="standard half-sum twin MSE; lambda min-twin half-MSE; head-stopped Huber discrepancy",
                reduction="global genuine-row mean including terminal; all five weights one",
                replay="uniform complete natural episodes with replacement; online/EMA full BPTT from zero",
                collection="uniform legal prefill then insert-one/update-one; carry/action reset only at done",
                update="one preupdate gradient snapshot; five Adam applications then all EMA",
                auxiliary_heads="two lambda twins only; no GVD parameters, RNG draws or objectives",
                precision="float32; x64 disabled; default matmul precision",
                environment="pinned upstream native5; sparse rewards; max_steps_in_episode=1000; no normalization")


class Learner(NamedTuple):
    online: dict
    target: dict
    optimizers: dict
    updates: jax.Array


class State(NamedTuple):
    learner: Learner
    replay: Any
    env_state: Any
    actor_carry: jax.Array
    critic_carry: Any
    previous_action: jax.Array
    rng: jax.Array
    prefill: jax.Array
    interactions: jax.Array
    completed_episodes: jax.Array
    evaluation_interactions: jax.Array
    healthy: jax.Array


@dataclass(frozen=True)
class Model:
    fe_graph: Any
    head_graph: Any
    critic_graph: Any
    width: int
    settings: Settings = Settings()
    actions: int = 25

    def encode(self, parameters, carry, observation, previous_action):
        return nnx.merge(self.fe_graph, parameters)(carry, observation[..., :1], previous_action)

    def head(self, parameters, latent):
        return nnx.merge(self.head_graph, parameters)(latent)

    def twins(self, parameters, latent):
        return jnp.stack([nnx.merge(self.critic_graph, p)(latent) for p in parameters], -1)

    def unroll(self, parameters, batch):
        initial = (jnp.zeros((batch.actions.shape[1], self.width), jnp.float32),
                   jnp.zeros((batch.actions.shape[1], self.actions), jnp.float32))

        def step(state, row):
            carry, previous = state
            observation, action, done, valid = row
            new_carry, latent = self.encode(parameters, carry, observation, previous)
            previous = jax.nn.one_hot(action, self.actions)
            reset = done | ~valid
            return (jnp.where(reset[:, None], 0, new_carry),
                    jnp.where(reset[:, None], 0, previous)), jnp.where(valid[:, None], latent, 0)

        return jax.lax.scan(step, initial,
                            (batch.observations, batch.actions, batch.dones, batch.valid))[1]


def agent_key(seed):
    if type(seed) is not int or seed not in range(10):
        raise ValueError("seed index outside 0..9")
    return jax.random.split(jax.random.PRNGKey(2029), 10)[seed]


def optimizer(model, name):
    s = model.settings
    return optax.adam(s.fe_learning_rate if name == "actor_fe" else s.head_learning_rate,
                      b1=s.b1, b2=s.b2, eps=s.eps, eps_root=s.eps_root)


def initialize_model(seed=0, *, width=512):
    """Width override is a deterministic test seam, not a training CLI setting."""
    fe_key, heads_key = jax.random.split(agent_key(seed))
    fe = RecurrentFeatureExtractor((1,), battleship_projection(width), "gru", width,
                                   25, rngs=nnx.Rngs(fe_key))
    graph, fe_params = nnx.split(fe)
    heads = [Head(width, (width,), 25, layer_norm=False, rngs=nnx.Rngs(k))
             for k in jax.random.split(heads_key, 7)]
    actor_graph, actor = nnx.split(heads[0])
    for head in heads[1:]:
        # Add deterministic LN after initialization, preserving every Linear draw.
        head.norms = [nnx.LayerNorm(width, epsilon=1e-6, dtype=jnp.float32,
                      use_fast_variance=True, rngs=nnx.Rngs(0))]
    pairs = [nnx.split(h) for h in heads[1:]]
    online = dict(actor_fe=fe_params, actor=actor,
                  standard=TwinCriticState(pairs[0][1], pairs[1][1]),
                  lambda1=TwinCriticState(pairs[2][1], pairs[3][1]),
                  lambda2=TwinCriticState(pairs[4][1], pairs[5][1]))
    model = Model(graph, actor_graph, pairs[0][0], width)
    return model, Learner(online, jax.tree.map(lambda x: jnp.array(x, copy=True), online),
                          {k: optimizer(model, k).init(p) for k, p in online.items()}, jnp.int32(0))


def policy(logits, legal):
    legal = legal.astype(bool)
    legal = legal | ~jnp.any(legal, axis=-1, keepdims=True)
    masked = jnp.where(legal, logits, -1e9)
    return jax.nn.softmax(masked, axis=-1), jax.nn.log_softmax(masked, axis=-1)


def selected(values, actions):
    return jnp.take_along_axis(values, actions[..., None], axis=-1)[..., 0]


def valid_mean(values, valid):
    return jnp.sum(jnp.where(valid, values, 0)) / jnp.maximum(valid.sum(), 1)


def lambda_targets(batch, q, probabilities, settings):
    """Q-conditioned lambda G: correction uses NEXT executed action/likelihood."""
    valid = batch.valid.astype(bool)
    q = jnp.where(valid[..., None, None], q, 0)
    pi = jnp.where(valid[..., None], probabilities, 0)
    actions = jnp.where(valid, batch.actions, 0)
    values = jnp.sum(pi[..., None] * q, axis=-2)
    qa = jnp.stack([selected(q[..., j], actions) for j in range(2)], -1)
    ratio = selected(pi, actions) / jnp.where(valid, batch.behavior, 1)
    c = jnp.asarray(settings.lambdas) * jnp.minimum(settings.c_bar, ratio[..., None])
    shift = lambda x: jnp.concatenate((x[1:], jnp.zeros_like(x[:1])), axis=0)
    rewards = jnp.where(valid, batch.rewards, 0)
    continuing = valid & ~batch.dones.astype(bool)

    def step(future, row):
        reward, active, vn, cn, qn = row
        value = reward[..., None] + settings.gamma * active[..., None] * (vn + cn * (future - qn))
        return value, value

    return stop(jax.lax.scan(step, jnp.zeros_like(values[0]),
                (rewards, continuing, shift(values), shift(c), shift(qa)), reverse=True)[1])


def targets(model, target, batch):
    target = stop(target)
    z = model.unroll(target["actor_fe"], batch)
    pi, _ = policy(model.head(target["actor"], z), batch.observations[..., 1:])
    q = [jnp.min(model.twins(target[n], z), -1) for n in ("standard", "lambda1", "lambda2")]
    # Keep the original standard bootstrap reduction/order, with no entropy term.
    values = jnp.stack([jnp.sum(pi * v, -1) for v in q], -1)
    values = jnp.where(batch.valid[..., None], values, 0)
    rewards = jnp.where(batch.valid, batch.rewards, 0)
    dones = batch.dones | ~batch.valid
    next_v = jnp.concatenate((values[1:, :, 0], jnp.zeros_like(values[:1, :, 0])), axis=0)
    standard = stop(rewards + model.settings.gamma * (~dones) * next_v)
    auxiliary = lambda_targets(batch, jnp.stack(q[1:], -1), pi, model.settings)
    ratios = jnp.where(batch.valid, selected(pi, batch.actions), 1) / jnp.where(batch.valid, batch.behavior, 1)
    return jnp.concatenate((standard[..., None], auxiliary), -1), stop(ratios)


def objectives(model, online, target, batch):
    if set(online) != set(GROUPS) or set(target) != set(GROUPS):
        raise ValueError("exactly five active parameter groups required")
    # The active producer has separate recurrent forwards for standard/LD and
    # actor/lambda TD. Sharing one z changes float32 FE gradient accumulation.
    base_z = model.unroll(online["actor_fe"], batch)
    z = model.unroll(online["actor_fe"], batch)
    pi, logpi = policy(model.head(online["actor"], z), batch.observations[..., 1:])
    goal, ratios = targets(model, target, batch)
    q = model.twins(online["standard"], base_z)
    executed = jnp.stack([selected(q[..., i], batch.actions) for i in range(2)], -1)
    teacher = stop(jnp.min(model.twins(online["standard"], z), -1))
    mean = lambda x: valid_mean(x, batch.valid)
    actor = mean(jnp.sum(pi * (model.settings.alpha * logpi - teacher), -1))
    standard = mean(jnp.mean(jnp.square(executed - goal[..., :1]), -1))
    auxiliary = []
    for j, name in enumerate(("lambda1", "lambda2")):
        prediction = selected(jnp.min(model.twins(online[name], z), -1), batch.actions)
        auxiliary.append(mean(.5 * (prediction - goal[..., j + 1])**2))
    ld = [selected(jnp.min(model.twins(stop(online[n]), base_z), -1), batch.actions)
          for n in ("lambda1", "lambda2")]
    discrepancy = mean(optax.huber_loss(ld[0], ld[1], delta=model.settings.huber_delta))
    terms = jnp.stack((jnp.zeros_like(actor), standard, jnp.zeros_like(actor),
                       jnp.zeros_like(actor), discrepancy))
    terms = terms.at[0].set(actor)
    for j, value in enumerate(auxiliary):
        terms = terms.at[j + 2].set(value)
    metrics = dict(zip(OBJECTIVES, terms))
    metrics.update(entropy=mean(-jnp.sum(pi * logpi, -1)),
        online_q_mean=mean(jnp.min(executed, -1)), standard_target_mean=mean(goal[..., 0]),
        lambda1_target_mean=mean(goal[..., 1]), lambda2_target_mean=mean(goal[..., 2]),
        discrepancy_abs=mean(jnp.abs(ld[0] - ld[1])), ratio_mean=mean(ratios),
        ratio_max=jnp.max(jnp.where(batch.valid, ratios, 0)),
        rho_clipped_fraction=mean(ratios > model.settings.rho_bar),
        c_clipped_fraction=mean(ratios > model.settings.c_bar),
        ratio_ess=jnp.square(jnp.sum(jnp.where(batch.valid, ratios, 0))) /
                  jnp.maximum(jnp.sum(jnp.where(batch.valid, ratios**2, 0)), 1e-20))
    return terms, metrics


def update(model, learner, batch):
    def loss(params):
        terms, metrics = objectives(model, params, learner.target, batch)
        return terms.sum(), metrics
    (value, metrics), gradients = jax.value_and_grad(loss, has_aux=True)(learner.online)
    online, opts = {}, {}
    for name, parameters in learner.online.items():
        delta, opts[name] = optimizer(model, name).update(gradients[name], learner.optimizers[name], parameters)
        online[name] = optax.apply_updates(parameters, delta)
    tau = model.settings.tau
    target = jax.tree.map(lambda new, old: tau * new + (1 - tau) * old, online, learner.target)
    return Learner(online, target, opts, learner.updates + 1), {**metrics, "loss": value}


class EnvState(NamedTuple):
    env_state: Any
    observation: jax.Array
    episode_id: jax.Array
    timestep: jax.Array


class Environment:
    """Retain upstream returned hit observation and track true episode boundaries."""
    def __init__(self, raw_env):
        self.raw_env = raw_env
        self.default_params = raw_env.default_params

    def reset(self, key, params):
        obs, state = self.raw_env.reset(key, params)
        return obs, EnvState(state, obs, jnp.int32(0), jnp.int32(0))

    def step(self, key, state, action, params):
        obs, raw, reward, done, info = self.raw_env.step(key, state.env_state, action, params)
        return obs, EnvState(raw, obs, state.episode_id + done.astype(jnp.int32),
                             jnp.where(done, 0, state.timestep + 1)), reward, done, info


@lru_cache(maxsize=1)
def environment():
    path = Path(__file__).resolve().parents[2] / "vendor/lambda_discrepancy/lamb/envs/battleship.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != "db597c2373931508ae063f45c779fbdef5fdd1e657c833b1aa890a141e9f7818":
        raise ValueError("pinned Battleship source changed")
    spec = importlib.util.spec_from_file_location("ac_lambda_battleship", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return Environment(module.Battleship(rows=5, cols=5, ship_lengths=(3, 2), dense_reward=False))


def initial_state(model, learner, seed=0, *, capacity=200000):
    env = environment()
    key = jax.random.fold_in(agent_key(seed), 1)
    key, reset = jax.random.split(key)
    _, state = env.reset(reset, env.default_params)
    zero = jnp.int32(0)
    return env, State(learner, replay.empty_replay(capacity, actions=25), state,
        jnp.zeros((1, model.width), jnp.float32), None, jnp.zeros((1, 25), jnp.float32),
        key, zero, zero, zero, zero, jnp.asarray(True))


def collect(model, env, state, *, uniform):
    rng, action_key, env_key, replay_key = jax.random.split(state.rng, 4)
    observation = state.env_state.observation
    carry, latent = model.encode(state.learner.online["actor_fe"], state.actor_carry,
                                  observation[None], state.previous_action)
    logits = jnp.zeros((1, model.actions), jnp.float32) if uniform else model.head(state.learner.online["actor"], latent)
    probabilities, logp = policy(logits, observation[None, 1:])
    action = jax.random.categorical(action_key, logp[0]).astype(jnp.int32)
    behavior = (1.0 / observation[1:].sum()) if uniform else probabilities[0, action]
    _, next_env, reward, done, _ = env.step(env_key, state.env_state, action, env.default_params)
    version = jnp.asarray(-1, jnp.int32) if uniform else state.learner.updates
    natural = done & (reward == model.settings.terminal_reward)
    buffer = replay.insert(state.replay, observation, action, reward, done, behavior, version,
                           state.env_state.episode_id, state.env_state.timestep, natural)
    healthy = (state.healthy & observation[1:][action].astype(bool) & (~done | natural)
               & buffer.pending_ok & (buffer.pending_length < model.actions) & (buffer.rejected == 0))
    return state._replace(replay=buffer, env_state=next_env, rng=rng,
        actor_carry=jnp.where(done, 0, carry),
        previous_action=jnp.where(done, 0, jax.nn.one_hot(action, model.actions)[None]),
        prefill=state.prefill + int(uniform), interactions=state.interactions + int(not uniform),
        completed_episodes=state.completed_episodes + done, healthy=healthy), replay_key


def training_step(model, env, state):
    state, key = collect(model, env, state, uniform=False)
    buffer, batch, metrics = replay.sample(state.replay, key, model.settings.batch_episodes, 25)
    state = state._replace(replay=buffer)
    metrics["policy_age_mean"] = valid_mean(state.learner.updates - batch.versions, batch.valid)
    learner, losses = update(model, state.learner, batch)
    healthy = state.healthy & batch.integrity & jnp.all(jnp.isfinite(jnp.stack(list(losses.values()))))
    learner = jax.lax.cond(healthy, lambda: learner, lambda: state.learner)
    return state._replace(learner=learner, healthy=healthy), {**metrics, **losses}


def kernels(model, env):
    def prefill(state, steps):
        return jax.lax.fori_loop(0, steps, lambda _, s: collect(model, env, s, uniform=True)[0], state)

    def train(state, steps):
        if not 1 <= steps <= 10000:
            raise ValueError("block outside 1..10000")
        return jax.lax.fori_loop(1, steps, lambda _, x: training_step(model, env, x[0]),
                                training_step(model, env, state))
    return jax.jit(prefill, static_argnums=(1,)), jax.jit(train, static_argnums=(1,))


def evaluation_keys(seed=0, episodes=500):
    key = jax.random.fold_in(agent_key(seed), 2)
    fold = jax.vmap(jax.random.fold_in, in_axes=(None, 0))
    indices = jnp.arange(episodes, dtype=jnp.uint32)
    return (fold(jax.random.fold_in(key, 0xE24), indices),
            fold(jax.random.fold_in(key, 0xA24), indices))


def evaluate(model, env, online, reset_keys, action_keys):
    observations, env_states = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env.default_params)
    n = reset_keys.shape[0]
    initial = (env_states, observations, jnp.zeros((n, model.width), jnp.float32),
               jnp.zeros((n, model.actions), jnp.float32), jnp.zeros(n, bool))

    def step(state, t):
        states, obs, carry, previous, completed = state
        carry, z = model.encode(online["actor_fe"], carry, obs, previous)
        _, logp = policy(model.head(online["actor"], z), obs[..., 1:])
        keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(action_keys, t)
        actions = jax.vmap(jax.random.categorical)(keys, logp).astype(jnp.int32)
        step_keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(reset_keys, t + 1)
        next_obs, states, rewards, dones, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            step_keys, states, actions, env.default_params)
        active = ~completed
        legal = selected(obs[..., 1:], actions).astype(bool)
        return (states, next_obs, jnp.where(dones[:, None], 0, carry),
                jnp.where(dones[:, None], 0, jax.nn.one_hot(actions, model.actions)), completed | dones), (
                jnp.where(active, rewards, 0), active, active & dones, ~active | legal)

    _, (rewards, active, done, legal) = jax.lax.scan(step, initial, jnp.arange(model.actions))
    return dict(returns=rewards.sum(axis=0), lengths=active.sum(axis=0),
                completed=done.any(axis=0), legal=legal.all(axis=0))
