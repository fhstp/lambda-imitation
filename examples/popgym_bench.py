"""Full POPGym benchmark on one multi-GPU host: one command, autonomous.

    python examples/popgym_bench.py --output /scratch/popgym-bench

Trains every discrete POPGym task (42 = 14 families x Easy/Medium/Hard) with:

- FlashSAC-inspired networks (residual RMS actor128/critic256 heads, two
  expansion-4 blocks, critic OFN, all six critics 51-atom categorical on
  [-1.5, 1.5], dense deterministic categorical projection), GRU512 memory with
  a 128-unit prev-action projection, complete-episode BPTT, reward-only
  projected main law and two-hot scalar-Retrace lambda heads, detached-head
  scalar lambda discrepancy (beta=10), actor-to-FE gradient stopped;
- lambda1=.05, lambda2=.99, all LRs 1e-4, alpha 1->.01 over 500k then fixed;
- correlated exploration: a unit-Gaussian logit bias held for k ~ p(k)~k^-2,
  k<=16 decisions, fresh categorical draw each step, exact conditional
  behaviour probability softmax(logits+bias)[a] stored for Retrace;
- 3 training seeds per task, genuinely jax.vmap-batched (training, prefill,
  replay, optimizer moments, rollouts and RNG are per-lane).

The supervisor schedules all PRIMARY tasks over the GPUs first (one worker
per GPU, longest expected first), and only after every primary task has finished it runs the
beta=0 CONTROL arm (identical otherwise). Every task resumes from its own
checkpoint if the script is restarted. It also serves the dashboard.
"""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STUDY = 'popgym-b200-flash-bench'
DEFAULTS = dict(
    n_seeds=3, seed_base=42, max_steps=1_000_000, lambda_coef=10., fe_lr=1e-4, actor_lr=1e-4, critic_lr=1e-4,
    alpha_initial=1., alpha_final=.01, alpha_decay_steps=500_000, tau=.005, gamma=.99, lambda1=.05, lambda2=.99,
    memory_hidden_dim=512, projection_dim=128, actor_width=128, critic_width=256, online_buffer_size=100_000,
    prefill_steps=12_800, batch_transitions=8320, min_batch_episodes=8, collect_steps=16, chunk_steps=2048,
    eval_every=20_000, checkpoint_every=100_000, eval_episodes=50, final_episodes=200,
    noise_strength=1., noise_power=2., noise_max_duration=16)
ARMS = {'primary': {}, 'control': {'lambda_coef': 0.}}
NOISE_RNG_TAG = 1464689751
VALIDATION_TASKS = ('popgym-BattleshipEasy-v0', 'popgym-AutoencodeEasy-v0')


def resolve_config(values):
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    c = {**DEFAULTS, **values}
    if (c['max_steps'] % c['eval_every'] or c['checkpoint_every'] % c['eval_every'] or c['eval_every'] % c['collect_steps']
            or c['chunk_steps'] % c['collect_steps']):
        raise ValueError('max_steps/eval_every/chunk_steps must be multiples of collect_steps (and eval_every)')
    if not 0 <= c['lambda1'] < c['lambda2'] <= 1 or not 0 < c['gamma'] < 1:
        raise ValueError('Invalid discount/lambdas')
    return c


def task_settings(c, horizon):
    """Constant transition budget per update: full episodes of length <= horizon.

    Battleship Easy (H=64) keeps the original 128 episodes x 65 slots."""
    return dict(batch_episodes=max(c['min_batch_episodes'], c['batch_transitions'] // (horizon + 1)))


def alpha_at(step, c):
    import jax.numpy as jnp
    t = jnp.clip(jnp.asarray(step) / c['alpha_decay_steps'], 0., 1.)
    return jnp.exp((1 - t) * jnp.log(jnp.float32(c['alpha_initial'])) + t * jnp.log(jnp.float32(c['alpha_final'])))


def agent_config(c, port, seed, small=False):
    s = task_settings(c, port.horizon)
    return dict(batch_size=s['batch_episodes'], online_buffer_size=c['online_buffer_size'], fe_lr=c['fe_lr'],
        actor_lr=c['actor_lr'], critic_lr=c['critic_lr'], alpha=c['alpha_initial'], gamma=c['gamma'], tau=c['tau'],
        burn_in_length=0, sequence_length=port.horizon, lambda_truncation=1, lambda1=c['lambda1'], lambda2=c['lambda2'],
        lambda_coef=c['lambda_coef'], projection_dim=c['projection_dim'], memory_type='gru',
        memory_hidden_dim=c['memory_hidden_dim'], head_dim=8 if small else 512, critic_layer_norm=False, use_sac=False,
        seed=seed, residual_actor_width=c['actor_width'], residual_critic_width=c['critic_width'])


def build(c, port, small=False):
    """Seed-stacked agent states plus a vmappable runner for ONE task."""
    import jax
    import jax.numpy as jnp
    from lambda_imitation.buffer import create_buffer
    import popgym_flash_combined as combined
    import popgym_flash_prototypes as flash
    from popgym_critic_aligned import replace_captures
    from battleship_sweep import deterministic_projection
    clone = flash.base.normalization.clone_globals
    seeds = list(range(c['seed_base'], c['seed_base'] + c['n_seeds']))
    states = []
    for seed in seeds:
        state, fns, debug = combined.networks(agent_config(c, port, seed, small), port)
        states.append(state)
    batch = task_settings(c, port.horizon)['batch_episodes']
    two_hot = lambda values: deterministic_projection(jnp.ones((values.size, 1)), values.reshape(-1, 1))
    objective = clone(flash.categorical_objective, {'project_distribution': deterministic_projection, 'two_hot': two_hot})
    sampler = flash.base.make_sampler(batch, port.horizon)
    loss, components, context = objective(fns, debug, sampler, 'all-categorical')
    update_original = replace_captures(fns.update, {'loss_combined': loss})

    @jax.jit
    def update(state, key):
        a = alpha_at(state.update_step, c)
        state, metrics = update_original(state._replace(alpha=a, log_alpha=jnp.log(a)), key)
        a = alpha_at(state.update_step, c)
        return state._replace(alpha=a, log_alpha=jnp.log(a)), metrics

    fns = fns._replace(update=update)
    _, bfns = create_buffer({k: v.shape[1:] for k, v in states[0].online_buffer.info.items()}, c['online_buffer_size'], 1, [], [])
    carry_dim = context['feature_extractor'].initialize_carry(1).shape[-1]
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    cls = HostRunner if getattr(port, 'is_host', False) else BenchRunner
    return stacked, cls(c, port, fns, debug, bfns, carry_dim), seeds


# ---------------------------------------------------------------- exploration
def noise_init(seed, action_dim):
    import jax
    import jax.numpy as jnp
    return (jax.random.fold_in(jax.random.PRNGKey(seed), NOISE_RNG_TAG), jnp.zeros(action_dim, jnp.float32),
            jnp.int32(0), jnp.int32(0))


def noise_advance(noise, c):
    """Bias for THIS decision; remaining = number of future reuses."""
    import jax
    import jax.numpy as jnp
    key, bias, remaining, duration = noise
    def refresh(_):
        k, bk, dk = jax.random.split(key, 3)
        weights = jnp.arange(1, c['noise_max_duration'] + 1, dtype=jnp.float32) ** (-c['noise_power'])
        length = (1 + jax.random.categorical(dk, jnp.log(weights / weights.sum()))).astype(jnp.int32)
        return k, jax.random.normal(bk, bias.shape, jnp.float32), length - 1, length
    return jax.lax.cond(remaining == 0, refresh, lambda _: (key, bias, remaining - 1, duration), None)


def noise_reset(noise, done):
    """Drop the held bias at real episode ends; the noise RNG keeps advancing."""
    import jax.numpy as jnp
    key, bias, remaining, duration = noise
    return key, jnp.where(done, 0., bias), jnp.where(done, 0, remaining), jnp.where(done, 0, duration)


def policy_logits_fn(fns):
    from flax import nnx
    from popgym_gradient_balance import captures
    cap = captures(fns.predict)
    if cap.get('mask_fn') is not None:
        raise ValueError('No action masks in this benchmark')
    fe_graph, run_actor, obs_fn = cap['feature_extractor_graph'], cap['run_actor'], cap['obs_fn']
    def logits(sac, obs, carry, prev):
        nc, x = nnx.merge(fe_graph, sac.feature_extractor)(carry[None], obs_fn(obs)[None], prev[None])
        return run_actor(sac.actor, x)[0], nc[0]
    return logits


def behaviour_step(logits_fn, sac, obs, carry, prev, noise, key, c, random, action_dim):
    """One acting decision: action, behaviour prob of the executed action, carry, noise."""
    import jax
    import jax.numpy as jnp
    if random:
        return jax.random.randint(key, (), 0, action_dim), jnp.float32(1. / action_dim), carry, noise
    logits, nc = logits_fn(sac, obs, carry, prev)
    if c['noise_strength']:
        noise = noise_advance(noise, c)
        logits = logits + c['noise_strength'] * noise[1]
    action = jax.random.categorical(key, logits)
    return action, jax.nn.softmax(logits)[action], nc, noise


def lane_cache():
    """jit(vmap over seed lanes) per (function, static trailing arguments)."""
    import jax
    cache = {}
    def get(fn, *static):
        key = (fn, *static)
        if key not in cache:
            cache[key] = jax.jit(jax.vmap(lambda *a: fn(*a, *static)))
        return cache[key]
    return get


def params_finite(states):
    """Device-side check of everything except the (large) replay buffer."""
    import jax
    import jax.numpy as jnp
    leaves = [x for x in jax.tree.leaves(states._replace(online_buffer=None)) if jnp.issubdtype(x.dtype, jnp.floating)]
    if not bool(jax.jit(lambda xs: jnp.all(jnp.stack([jnp.isfinite(x).all() for x in xs])))(leaves)):
        raise FloatingPointError('Nonfinite agent state')


def mark_boundary(sac, size):
    b = sac.online_buffer
    newest = (b.pos - 1) % size
    return sac._replace(online_buffer=b._replace(sampling_ok=b.sampling_ok.at[newest].set(False)))


def update_block(fns, sac, key, count):
    import jax
    keys = jax.random.split(key, count)
    sac, metrics = jax.lax.scan(lambda s, k: fns.update(s, k), sac, keys)
    return sac, jax.tree.map(lambda x: x.mean(), metrics)


def episode_outputs(debug, fns, sac, obs, carry, prev, active, key, mode, action_dim, step_env):
    """Shared per-step evaluation logic (no exploration noise)."""
    import jax
    import jax.numpy as jnp
    values, nc = debug.predict_values(sac, obs, carry, prev)
    probs, qs = values['probs'], values['q'].min(-1)
    if mode == 'random':
        action = jax.random.randint(key, (), 0, action_dim)
    elif mode == 'critic':
        action = jnp.argmax(qs)
    elif mode == 'greedy':
        action = jnp.argmax(probs)
    else:
        action = jax.random.categorical(key, jnp.log(probs + 1e-30))
    entropy = -(probs * jnp.log(probs + 1e-30)).sum()
    ld = values['lambda1_q'][action].min() - values['lambda2_q'][action].min()
    return action, nc, qs[action], entropy, ld


class BenchRunner:
    """Compiled collect/update blocks for JAX ports, explicit persistent rollout.

    rollout = (key, obs, env_state, carry, prev_action, overflow, age, noise)."""

    def __init__(self, c, port, fns, debug, bfns, carry_dim):
        import jax
        import jax.numpy as jnp
        self.c, self.port, self.fns, self.debug = c, port, fns, debug
        self.zero_carry = jnp.zeros(carry_dim, jnp.float32)
        self.zero_prev = jnp.zeros(port.action_dim, jnp.float32)
        logits_fn = policy_logits_fn(fns)
        A, H, size = port.action_dim, port.horizon, c['online_buffer_size']

        def collect(sac, rollout, random):
            key, obs, es, carry, prev, overflow, age, noise = rollout
            key, ak, ek = jax.random.split(key, 3)
            action, prob, nc, noise = behaviour_step(logits_fn, sac, obs, carry, prev, noise, ak, c, random, A)
            action = action.astype(jnp.int32)
            nobs, ns, reward, done = port.step(ek, es, action)
            t = {'observations': obs, 'actions': jnp.atleast_1d(action).astype(jnp.float32),
                 'rewards': jnp.asarray(reward, jnp.float32), 'terminated': done.astype(jnp.float32),
                 'behaviour_weight': prob}
            buffer = bfns.add(sac.online_buffer, t, done)
            nc = jnp.where(done, self.zero_carry, nc)
            np_ = jnp.where(done, self.zero_prev, fns.encode_action(jnp.atleast_1d(action)))
            overflow = overflow | ((age + 1 >= H) & ~done)
            age = jnp.where(done, 0, age + 1)
            if not random:
                noise = noise_reset(noise, done)
            return sac._replace(online_buffer=buffer), (key, nobs, ns, nc, np_, overflow, age, noise)

        def prefill(sac, rollout, n_steps):
            (sac, rollout), _ = jax.lax.scan(lambda cr, _: (collect(*cr, random=True), None), (sac, rollout), None, length=n_steps)
            def finish(cr):
                sac, r, extra = cr
                sac, r = collect(sac, r, random=True)
                return sac, r, extra + 1
            sac, rollout, extra = jax.lax.while_loop(lambda cr: (cr[1][6] != 0) & (cr[2] < H), finish, (sac, rollout, jnp.int32(0)))
            return mark_boundary(sac, size), rollout, n_steps + extra

        def block(sac, rollout):
            (sac, rollout), _ = jax.lax.scan(lambda cr, _: (collect(*cr, random=False), None), (sac, rollout), None, length=c['collect_steps'])
            sac = mark_boundary(sac, size)
            key, *rest = rollout
            key, uk = jax.random.split(key)
            sac, metrics = update_block(fns, sac, uk, c['collect_steps'])
            return sac, (key, *rest), metrics

        def train(sac, rollout, n_steps):
            def body(cr, _):
                s, r, m = block(*cr)
                return (s, r), m
            (sac, rollout), metrics = jax.lax.scan(body, (sac, rollout), None, length=n_steps // c['collect_steps'])
            return sac, rollout, jax.tree.map(lambda x: x.mean(), metrics)

        def evaluate(sac, seed, mode, n_episodes):
            def episode(index):
                key, rk = jax.random.split(jax.random.PRNGKey(seed + index))
                obs, es = port.reset(rk)
                def body(cr, _):
                    key, obs, es, carry, prev, active = cr
                    key, ak, ek = jax.random.split(key, 3)
                    action, nc, q, entropy, ld = episode_outputs(debug, fns, sac, obs, carry, prev, active, ak, mode, A, None)
                    nobs, nes, reward, done = port.transition(ek, es, action)
                    out = (jnp.where(active, reward, 0.), active, active & done, jnp.where(active, q, 0.),
                           jnp.where(active, entropy, 0.), jnp.where(active, ld, 0.))
                    return (key, nobs, nes, nc, fns.encode_action(jnp.atleast_1d(action)), active & ~done), out
                _, out = jax.lax.scan(body, (key, obs, es, self.zero_carry, self.zero_prev, jnp.bool_(True)), None, length=H)
                return out
            return jax.vmap(episode)(jnp.arange(n_episodes, dtype=jnp.uint32))

        lanes = lane_cache()
        self.prefill = lambda s, r, n: lanes(prefill, n)(s, r)
        self.train = lambda s, r, n: lanes(train, n)(s, r)
        self.evaluate = lambda s, seeds, mode, n: lanes(evaluate, mode, n)(s, seeds)

    def initial_rollout(self, seeds):
        import jax
        import jax.numpy as jnp
        def one(seed):
            key, rk = jax.random.split(jax.random.PRNGKey(seed))
            obs, state = self.port.reset(rk)
            return (key, obs, state, self.zero_carry, self.zero_prev, jnp.bool_(False), jnp.int32(0),
                    noise_init(seed, self.port.action_dim))
        return jax.vmap(one)(jnp.asarray(seeds, jnp.uint32))

    def check_rollout(self, rollout, at_boundary=False):
        import numpy as np
        if np.asarray(rollout[5]).any():
            raise RuntimeError('Episode exceeded the task horizon')
        if at_boundary and np.asarray(rollout[6]).any():
            raise RuntimeError('Prefill must end at a natural episode boundary')


class HostRunner:
    """Same agent/objective for host Gymnasium tasks (Labyrinth).

    Lanes step their own env on the host; acting, replay inserts and the
    16-update blocks are vmapped jitted device calls. rollout = dict with the
    device acting state (key, carry, prev, noise) and the host envs/obs."""

    def __init__(self, c, port, fns, debug, bfns, carry_dim):
        import jax
        import jax.numpy as jnp
        self.c, self.port, self.fns, self.debug = c, port, fns, debug
        self.zero_carry = jnp.zeros(carry_dim, jnp.float32)
        self.zero_prev = jnp.zeros(port.action_dim, jnp.float32)
        logits_fn = policy_logits_fn(fns)
        A, size = port.action_dim, c['online_buffer_size']

        def act(sac, obs, carry, prev, noise, key, random):
            key, ak = jax.random.split(key)
            action, prob, nc, noise = behaviour_step(logits_fn, sac, obs, carry, prev, noise, ak, c, random, A)
            return action.astype(jnp.int32), prob, nc, noise, key

        def after(nc, action, noise, done):
            nc = jnp.where(done, self.zero_carry, nc)
            prev = jnp.where(done, self.zero_prev, fns.encode_action(jnp.atleast_1d(action)))
            return nc, prev, noise_reset(noise, done)

        def add(buffer, t):
            return bfns.add(buffer, {k: v for k, v in t.items() if k != 'dones'}, t['dones'] > 0)

        def insert(sac, batch, valid):
            # Prefill only: lanes stop at their own natural boundary.
            def body(buffer, item):
                t, ok = item
                return jax.tree.map(lambda a, b: jnp.where(ok, a, b), add(buffer, t), buffer), None
            buffer, _ = jax.lax.scan(body, sac.online_buffer, (batch, valid))
            return mark_boundary(sac._replace(online_buffer=buffer), size)

        def insert_update(sac, batch, key):
            buffer, _ = jax.lax.scan(lambda b, t: (add(b, t), None), sac.online_buffer, batch)
            sac = mark_boundary(sac._replace(online_buffer=buffer), size)
            return update_block(fns, sac, key, c['collect_steps'])

        def eval_step(sac, obs, carry, prev, active, key, mode):
            # Lanes x episodes in one forward per time step.
            return jax.vmap(lambda *a: episode_outputs(debug, fns, sac, *a, mode, A, None))(obs, carry, prev, active, key)

        lanes = lane_cache()
        self.act = lambda sac, obs, carry, prev, noise, key, random: lanes(act, random)(sac, obs, carry, prev, noise, key)
        self.after = jax.jit(jax.vmap(after))
        self.insert = jax.jit(jax.vmap(insert))
        self.insert_update = jax.jit(jax.vmap(insert_update))
        self.eval_step = lambda sac, obs, carry, prev, active, key, mode: lanes(eval_step, mode)(sac, obs, carry, prev, active, key)

    def initial_rollout(self, seeds):
        import jax
        import jax.numpy as jnp
        import numpy as np
        envs, obs = [], []
        for lane, seed in enumerate(seeds):
            env = self.port.make()
            o, _ = env.reset(seed=int(seed))
            envs.append(env); obs.append(np.asarray(o, np.float32))
        n = len(seeds)
        return dict(key=jax.vmap(lambda s: jax.random.split(jax.random.PRNGKey(s))[0])(jnp.asarray(seeds, jnp.uint32)),
                    carry=jnp.tile(self.zero_carry, (n, 1)), prev=jnp.tile(self.zero_prev, (n, 1)),
                    noise=jax.vmap(lambda s: noise_init(s, self.port.action_dim))(jnp.asarray(seeds, jnp.uint32)),
                    envs=envs, obs=np.stack(obs), age=np.zeros(n, int))

    def _collect(self, sac, rollout, count, random, lanes_active=None):
        import jax
        import numpy as np
        n = len(rollout['envs'])
        rows = {k: [] for k in ('observations', 'actions', 'rewards', 'terminated', 'behaviour_weight', 'dones')}
        valid = []
        for _ in range(count):
            active = np.ones(n, bool) if lanes_active is None else lanes_active(rollout)
            action, prob, nc, noise, key = self.act(sac, jax.numpy.asarray(rollout['obs']), rollout['carry'],
                                                    rollout['prev'], rollout['noise'], rollout['key'], random)
            action_host = np.asarray(action)
            obs_next, rewards, dones = [], np.zeros(n, np.float32), np.zeros(n, bool)
            for i, env in enumerate(rollout['envs']):
                if not active[i]:
                    obs_next.append(rollout['obs'][i]); continue
                o, r, term, trunc, _ = env.step(int(action_host[i]))
                if not np.isfinite(r) or not np.all(np.isfinite(o)):
                    raise FloatingPointError('Nonfinite host reward/observation')
                rewards[i], dones[i] = r, term or trunc
                rollout['age'][i] = 0 if dones[i] else rollout['age'][i] + 1
                if not dones[i] and rollout['age'][i] >= self.port.horizon:
                    raise RuntimeError('Host episode exceeded the task horizon')
                if dones[i]:
                    o, _ = env.reset()
                obs_next.append(np.asarray(o, np.float32))
            rows['observations'].append(rollout['obs']); rows['actions'].append(action_host[:, None].astype(np.float32))
            rows['rewards'].append(rewards); rows['terminated'].append(dones.astype(np.float32))
            rows['behaviour_weight'].append(np.asarray(prob)); rows['dones'].append(dones.astype(np.float32))
            valid.append(active)
            # Inactive lanes keep their carry/noise/key-unaffected state.
            keep = jax.numpy.asarray(active)
            nc2, prev2, noise2 = self.after(nc, action, noise, jax.numpy.asarray(dones))
            sel = lambda new, old: jax.vmap(lambda k, a, b: jax.numpy.where(k, a, b))(keep, new, old)
            rollout['carry'], rollout['prev'] = sel(nc2, rollout['carry']), sel(prev2, rollout['prev'])
            rollout['noise'] = jax.tree.map(sel, noise2, rollout['noise'])
            rollout['key'] = key
            rollout['obs'] = np.stack(obs_next)
        batch = {k: np.swapaxes(np.stack(v), 0, 1) for k, v in rows.items()}
        return batch, np.swapaxes(np.stack(valid), 0, 1)

    def prefill(self, sac, rollout, n_steps):
        import jax.numpy as jnp
        import numpy as np
        counts = np.zeros(len(rollout['envs']), int)
        done_prefill = lambda r: (counts >= n_steps) & (r['age'] == 0)
        chunk = 1024
        while not done_prefill(rollout).all():
            def active(r):
                return ~done_prefill(r)
            batch, valid = self._collect(sac, rollout, chunk if (counts < n_steps).any() else 64, True, active)
            counts += valid.sum(1)
            sac = self.insert(sac, {k: jnp.asarray(v) for k, v in batch.items()}, jnp.asarray(valid))
        return sac, rollout, counts

    def train(self, sac, rollout, n_steps):
        import jax
        import jax.numpy as jnp
        metrics = []
        for _ in range(n_steps // self.c['collect_steps']):
            batch, _ = self._collect(sac, rollout, self.c['collect_steps'], False)
            keys = jax.vmap(jax.random.split)(rollout['key'])
            rollout['key'] = keys[:, 0]
            sac, m = self.insert_update(sac, {k: jnp.asarray(v) for k, v in batch.items()}, keys[:, 1])
            metrics.append(m)
        return sac, rollout, jax.tree.map(lambda *xs: jnp.mean(jnp.stack(xs), 0), *metrics)

    def evaluate(self, sac, seeds, mode, n_episodes):
        import jax
        import jax.numpy as jnp
        import numpy as np
        seeds = np.asarray(seeds)
        L, H = len(seeds), self.port.horizon
        envs = [[self.port.make() for _ in range(n_episodes)] for _ in range(L)]
        obs = np.stack([np.stack([np.asarray(envs[l][e].reset(seed=int(seeds[l] + e))[0], np.float32)
                                  for e in range(n_episodes)]) for l in range(L)])
        carry = jnp.tile(self.zero_carry, (L, n_episodes, 1)); prev = jnp.tile(self.zero_prev, (L, n_episodes, 1))
        active = np.ones((L, n_episodes), bool)
        keys = jax.vmap(lambda s: jax.vmap(lambda e: jax.random.PRNGKey(s + e))(jnp.arange(n_episodes, dtype=jnp.uint32)))(jnp.asarray(seeds, jnp.uint32))
        out = {k: np.zeros((L, n_episodes, H), np.float32 if k not in ('active', 'done') else bool)
               for k in ('reward', 'active', 'done', 'q', 'entropy', 'ld')}
        for t in range(H):
            if not active.any():
                break
            split = jax.vmap(jax.vmap(jax.random.split))(keys)
            keys, step_keys = split[:, :, 0], split[:, :, 1]
            action, nc, q, entropy, ld = self.eval_step(sac, jnp.asarray(obs), carry, prev, jnp.asarray(active), step_keys, mode)
            action, q, entropy, ld = map(np.asarray, (action, q, entropy, ld))
            for l in range(L):
                for e in range(n_episodes):
                    if not active[l, e]:
                        continue
                    o, r, term, trunc, _ = envs[l][e].step(int(action[l, e]))
                    out['reward'][l, e, t], out['active'][l, e, t] = r, True
                    out['q'][l, e, t], out['entropy'][l, e, t], out['ld'][l, e, t] = q[l, e], entropy[l, e], ld[l, e]
                    if term or trunc:
                        out['done'][l, e, t] = True; active[l, e] = False
                    obs[l, e] = np.asarray(o, np.float32)
            carry = nc
            prev = jax.nn.one_hot(jnp.asarray(action), self.port.action_dim, dtype=jnp.float32)
        for row in envs:
            for env in row:
                env.close()
        return tuple(out[k] for k in ('reward', 'active', 'done', 'q', 'entropy', 'ld'))

    def check_rollout(self, rollout, at_boundary=False):
        import numpy as np
        if at_boundary and np.asarray(rollout['age']).any():
            raise RuntimeError('Prefill must end at a natural episode boundary')


# ------------------------------------------------------------------- worker
def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False))
    tmp.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    base = HERE.parent
    paths = sorted((base / 'src/lambda_imitation').glob('*.py')) + sorted(base.glob('examples/popgym*.py')) + [base / 'examples/battleship_sweep.py']
    return {str(p.relative_to(base)): file_hash(p) for p in paths}


def runtime_versions():
    import importlib.metadata
    names = ('jax', 'jaxlib', 'flax', 'optax', 'numpy', 'cloudpickle', 'popjym', 'gymnax', 'gymnasium', 'popgym')
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def finite(tree):
    import jax
    import numpy as np
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(tree) if np.asarray(x).dtype.kind == 'f'):
        raise FloatingPointError('Nonfinite state/metrics')


def summarize_eval(outputs, mode, c, alpha):
    """Per-seed metrics from lane-stacked evaluation outputs."""
    import numpy as np
    from popgym_jax import evaluation_result
    args = SimpleNamespace(gamma=c['gamma'], alpha=alpha, use_sac=False)
    arrays = [np.asarray(x) for x in outputs]
    return [evaluation_result(tuple(a[lane] for a in arrays), mode, args, True)['metrics'] for lane in range(arrays[0].shape[0])]


def seed_stats(values):
    import numpy as np
    v = np.asarray(values, float)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0., n=len(v), seeds=[float(x) for x in v])


def task_folder(output, arm, env_id):
    return Path(output) / arm / env_id


def host_rollout(rollout):
    import jax
    if isinstance(rollout, dict):  # host envs pickle with their own RNG state
        return {k: (jax.device_get(v) if k in ('key', 'carry', 'prev', 'noise') else v) for k, v in rollout.items()}
    return jax.device_get(rollout)


def restore_rollout(saved):
    import jax
    import jax.numpy as jnp
    if isinstance(saved, dict):
        return {k: (jax.tree.map(jnp.asarray, v) if k in ('key', 'carry', 'prev', 'noise') else v) for k, v in saved.items()}
    return jax.tree.map(jnp.asarray, saved)


def save_checkpoint(path, payload):
    import cloudpickle
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        cloudpickle.dump(payload, f)
    tmp.replace(path)


def wandb_safe(fn, folder):
    """W&B publication failures must never kill training; keep a local trace."""
    try:
        return fn()
    except Exception as error:  # noqa: BLE001
        with (folder / 'wandb-errors.log').open('a') as f:
            f.write(f'{time.time()} {type(error).__name__}: {error}\n')
        return None


def train_task(output, arm, env_id, wandb_cfg, small=False):
    import cloudpickle
    import jax
    import jax.numpy as jnp
    import numpy as np
    from popgym_bench_envs import make_port, backend_for
    values = json.loads((Path(output) / 'study.json').read_text())['config'] if (Path(output) / 'study.json').exists() else {}
    c = resolve_config({**values, **ARMS[arm]})
    folder = task_folder(output, arm, env_id)
    folder.mkdir(parents=True, exist_ok=True)
    port = make_port(env_id)
    settings = task_settings(c, port.horizon)
    meta = dict(env_id=env_id, arm=arm, backend=backend_for(env_id), port_id=port.port_id, horizon=port.horizon,
                action_dim=port.action_dim, obs_shape=list(port.obs_shape), **settings)
    manifest = source_manifest()
    checkpoint = folder / 'latest.pkl'
    saved = None
    if checkpoint.exists():
        with checkpoint.open('rb') as f:
            saved = cloudpickle.load(f)
        if saved['config'] != c or saved['meta'] != meta:
            raise ValueError('Checkpoint configuration differs; use a new output directory')
        if saved['source_manifest'] != manifest:
            # Resuming with changed sources is recorded, not silently hidden.
            with (folder / 'resume-source-changes.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=time.time(), old=saved['source_manifest'], new=manifest)) + '\n')
    atomic_json(folder / 'config.json', dict(config=c, meta=meta, source_manifest=manifest, versions=runtime_versions(),
                                             devices=[str(d) for d in jax.devices()]))
    atomic_json(folder / 'status.json', dict(state='building', steps=saved['steps'] if saved else 0, target_steps=c['max_steps'], time=time.time()))
    states, runner, seeds = build(c, port, small)
    if saved:
        states = jax.tree.map(jnp.asarray, saved['state'])
        rollout = restore_rollout(saved['rollout'])
        step, history, interactions = saved['steps'], saved['history'], saved['interactions']
    else:
        rollout = runner.initial_rollout(seeds)
        atomic_json(folder / 'status.json', dict(state='prefill', steps=0, target_steps=c['max_steps'], time=time.time()))
        states, rollout, counts = jax.block_until_ready(runner.prefill(states, rollout, c['prefill_steps']))
        runner.check_rollout(rollout, at_boundary=True)
        params_finite(states)
        step, history, interactions = 0, [], [int(x) for x in np.asarray(counts)]

    def checkpoint_now():
        save_checkpoint(checkpoint, dict(state=jax.device_get(states), rollout=host_rollout(rollout), config=c, meta=meta,
            steps=step, history=history, interactions=interactions, source_manifest=manifest, wandb_id=run_id))
    run_id = hashlib.sha256(f'{STUDY}/{Path(output).resolve()}/{arm}/{env_id}'.encode()).hexdigest()[:16]
    run = None
    if wandb_cfg.get('project'):
        import wandb
        run = wandb_safe(lambda: wandb.init(project=wandb_cfg['project'], entity=wandb_cfg.get('entity'), id=run_id,
            resume='allow', group=env_id, job_type=arm, name=f'{env_id.removeprefix("popgym-").removesuffix("-v0")}-{arm}',
            tags=[arm, meta['backend'], env_id], config={**c, **meta, 'seeds': seeds}, dir=str(folder),
            settings=wandb.Settings(console='off')), folder)
        if run is not None:
            run.define_metric('steps')
            run.define_metric('*', step_metric='steps')
    log = (lambda payload: wandb_safe(lambda: run.log(payload), folder)) if run is not None else (lambda payload: None)

    def evaluate(final):
        n = c['final_episodes'] if final else c['eval_episodes']
        modes = ('stochastic', 'greedy', 'critic', 'random') if final else ('stochastic', 'greedy')
        base = 20_000_000 if final else 10_000_000
        lane_seeds = jnp.asarray([base + 10_000 * s for s in seeds], jnp.uint32)
        alpha = float(np.asarray(states.alpha)[0])
        row = dict(steps=step, final=final, time=time.time(), alpha=alpha, eval={})
        for mode in modes:
            outputs = jax.device_get(runner.evaluate(states, lane_seeds, mode, n))
            per_seed = summarize_eval(outputs, mode, c, alpha)
            row['eval'][mode] = {k: seed_stats([m[k] for m in per_seed]) for k in per_seed[0]}
        return row

    started, start_step = time.monotonic(), step
    try:
        if not history:
            history.append(evaluate(False))
            atomic_json(folder / 'history.json', history)
            checkpoint_now()
        next_eval = (step // c['eval_every'] + 1) * c['eval_every']
        while step < c['max_steps']:
            n = min(c['chunk_steps'], next_eval - step)
            states, rollout, metrics = runner.train(states, rollout, n)
            metrics = jax.device_get(metrics)
            finite(metrics)
            runner.check_rollout(rollout)
            step += n
            atomic_json(folder / 'status.json', dict(state='running', steps=step, target_steps=c['max_steps'], time=time.time(),
                steps_per_second=(step - start_step) / max(time.monotonic() - started, 1e-6)))
            if step < next_eval:
                continue
            next_eval += c['eval_every']
            row = evaluate(False)
            row['train'] = {k: seed_stats(np.asarray(v)) for k, v in metrics.items()}
            row['steps_per_second'] = (step - start_step) / max(time.monotonic() - started, 1e-6)
            history.append(row)
            params_finite(states)
            payload = {'steps': step, 'throughput/steps_per_second': row['steps_per_second']}
            for mode, metrics_ in row['eval'].items():
                for k, s in metrics_.items():
                    payload[f'eval/{mode}/{k}/mean'] = s['mean']; payload[f'eval/{mode}/{k}/std'] = s['std']
            for k, s in row['train'].items():
                payload[f'train/{k}/mean'] = s['mean']
            for seed, value in zip(seeds, row['eval']['stochastic']['return_mean']['seeds']):
                payload[f'seed/{seed}/eval/stochastic/return_mean'] = value
            log(payload)
            atomic_json(folder / 'history.json', history)
            if step % c['checkpoint_every'] == 0 or step == c['max_steps']:
                checkpoint_now()
        final = evaluate(True)
        history.append(final)
        atomic_json(folder / 'history.json', history)
        log({'steps': step, **{f'final/{m}/{k}/mean': s['mean'] for m, d in final['eval'].items() for k, s in d.items()},
             **{f'final/{m}/{k}/std': s['std'] for m, d in final['eval'].items() for k, s in d.items()}})
        if run is not None:
            wandb_safe(lambda: run.summary.update({'final_stochastic_return': final['eval']['stochastic']['return_mean']['mean'],
                                                   'final_greedy_return': final['eval']['greedy']['return_mean']['mean']}), folder)
        atomic_json(folder / 'status.json', dict(state='complete', steps=step, target_steps=c['max_steps'], time=time.time()))
    except BaseException:
        atomic_json(folder / 'status.json', dict(state='failed', steps=step, target_steps=c['max_steps'], time=time.time(),
                                                 error=traceback.format_exc()[-4000:]))
        raise
    finally:
        if run is not None:
            wandb_safe(lambda: run.finish(), folder)


# --------------------------------------------------------------- supervisor
def detect_gpus():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible:
        return [g for g in visible.split(',') if g.strip()]
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True)
        return [line.strip() for line in out.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError):
        return []


def task_order(env_ids):
    """Longest expected first (LPT), to shorten the tail of the schedule."""
    return sorted(env_ids, key=expected_hours, reverse=True)


def expected_hours(env_id):
    """Rough local-H100 hours per 1M steps x 5 seeds, fitted to measured
    throughput (CartPole 17, RepeatFirstHard 28, Battleship Easy/Hard 26/40)."""
    from popgym_bench_envs import make_port
    port = make_port(env_id)
    return 16. + .014 * port.horizon + .15 * port.action_dim


def validate(output, env_ids, updates=64, small=False):
    """Launch gate on the real device, discarded afterwards: same-input
    repeatability, seed-lane isolation and host round-trip resume, all exact."""
    import cloudpickle
    import jax
    import numpy as np
    from battleship_sweep import exact
    from popgym_bench_envs import make_port
    c = resolve_config(json.loads((Path(output) / 'study.json').read_text())['config'])
    results = {}
    for env_id in env_ids:
        port = make_port(env_id)
        states, runner, seeds = build(c, port, small)
        rollout = runner.initial_rollout(seeds)
        states, rollout, _ = jax.block_until_ready(runner.prefill(states, rollout, c['prefill_steps']))
        runner.check_rollout(rollout, at_boundary=True)
        start = time.monotonic()
        first = jax.block_until_ready(runner.train(states, rollout, updates))
        exact(first, jax.block_until_ready(runner.train(states, rollout, updates)))
        info = dict(states.online_buffer.info)
        info['rewards'] = info['rewards'].at[0].add(1.)
        changed = states._replace(online_buffer=states.online_buffer._replace(info=info))
        other = jax.block_until_ready(runner.train(changed, rollout, updates))
        if c['n_seeds'] > 1:
            exact(jax.tree.map(lambda x: x[1:], first), jax.tree.map(lambda x: x[1:], other))
            if all(np.array_equal(np.asarray(a)[0], np.asarray(b)[0])
                   for a, b in zip(jax.tree.leaves(first[0].critic), jax.tree.leaves(other[0].critic))):
                raise AssertionError('Perturbed lane did not change')
        restored = jax.tree.map(jax.numpy.asarray, cloudpickle.loads(cloudpickle.dumps(jax.device_get(first[:2]))))
        exact(jax.block_until_ready(runner.train(first[0], first[1], updates)),
              jax.block_until_ready(runner.train(*restored, updates)))
        params_finite(first[0])
        np.testing.assert_array_equal(np.asarray(first[0].update_step), np.full(c['n_seeds'], updates))
        results[env_id] = dict(passed=True, seconds=time.monotonic() - start, horizon=port.horizon, **task_settings(c, port.horizon))
    proof = dict(passed=True, updates=updates, tasks=results, device=str(jax.devices()[0]), device_kind=jax.devices()[0].device_kind,
                 backend=jax.default_backend(), source_manifest=source_manifest(), versions=runtime_versions(),
                 checks='same-input repeatability, lane isolation, host round-trip resume: exact bytes',
                 note='Validation states are discarded; production starts from fresh initialization and prefill.')
    atomic_json(Path(output) / 'validation.json', proof)
    return proof


def read_status(folder):
    try:
        return json.loads((folder / 'status.json').read_text())
    except (OSError, ValueError):
        return {}


def supervise(args):
    # Workers get the original environment; the supervisor itself must never
    # initialize (and preallocate) a GPU while reading task metadata.
    worker_env = dict(os.environ)
    os.environ['JAX_PLATFORMS'] = 'cpu'
    from popgym_bench_envs import all_env_ids
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    study = output / 'study.json'
    config = resolve_config(json.loads(args.config.read_text()) if args.config else {})
    record = dict(study=STUDY, config=config, arms=ARMS, wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
                  envs=args.envs or all_env_ids(), created=time.time())
    if study.exists():
        old = json.loads(study.read_text())
        if old['config'] != config or old['envs'] != record['envs']:
            raise ValueError(f'{study} has a different configuration/task list; use a new --output')
        record = old
    else:
        atomic_json(study, record)
    order = task_order(record['envs'])
    gpus = args.gpus or detect_gpus()
    if not gpus:
        raise RuntimeError('No GPUs detected; pass --gpus')
    slots = [g for g in gpus for _ in range(args.workers_per_gpu)]
    print(f'{len(order)} tasks x {len(ARMS)} arms on GPUs {gpus} ({args.workers_per_gpu} workers each)', flush=True)
    if args.dashboard_port:
        start_dashboard(output, args.dashboard_host, args.dashboard_port)
    proof = output / 'validation.json'
    current = source_manifest()
    if not args.skip_validation and not (proof.exists() and json.loads(proof.read_text()).get('source_manifest') == current):
        print(f'Validation gate on GPU {gpus[0]}: {", ".join(VALIDATION_TASKS)}', flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), 'validate', '--output', str(output)]
        env = {**worker_env, 'CUDA_VISIBLE_DEVICES': gpus[0], 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'}
        with open(output / 'validation.log', 'a') as log:
            code = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE.parent))
        if code:
            raise SystemExit(f'Validation gate FAILED (see {output / "validation.log"}); nothing launched')
        print('Validation passed.', flush=True)
    running, attempts = {}, {}
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    stop = False
    while True:
        for key, (proc, slot) in list(running.items()):
            if proc.poll() is not None:
                del running[key]
                arm, env_id = key
                state = read_status(task_folder(output, arm, env_id)).get('state')
                print(f'[{time.strftime("%H:%M:%S")}] {arm}/{env_id} exited {proc.returncode} ({state})', flush=True)
        def pending(arm):
            result = []
            for env_id in order:
                key = (arm, env_id)
                if key in running:
                    continue
                state = read_status(task_folder(output, arm, env_id)).get('state')
                if state == 'complete':
                    continue
                # One automatic resume from the last checkpoint after a crash
                # (e.g. transient I/O); repeated failures stay failed.
                if attempts.get(key, 0) >= (1 + args.max_resumes):
                    continue
                result.append(key)
            return result
        primary = pending('primary')
        primary_busy = primary or any(k[0] == 'primary' for k in running)
        queue = primary if primary_busy else pending('control') if not args.skip_control else []
        atomic_json(output / 'scheduler.json', dict(time=time.time(), gpus=gpus, slots=len(slots),
            phase='primary' if primary_busy else 'control', queued=[list(k) for k in queue],
            running=[dict(arm=k[0], env_id=k[1], gpu=slot) for k, (_, slot) in running.items()],
            attempts=[dict(arm=k[0], env_id=k[1], attempts=v) for k, v in attempts.items()]))
        if not queue and not running:
            break
        busy = [slot for _, slot in running.values()]
        free = [g for g in slots]
        for g in busy:
            free.remove(g)
        for key in queue:
            if not free or stop:
                break
            gpu = free.pop(0)
            arm, env_id = key
            attempts[key] = attempts.get(key, 0) + 1
            log = (logs / f'{arm}-{env_id}.log').open('a')
            env = {**worker_env, 'CUDA_VISIBLE_DEVICES': gpu, 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
                   'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
            cmd = [sys.executable, str(Path(__file__).resolve()), 'worker', '--output', str(output), '--arm', arm, '--env', env_id]
            if args.wandb_project:
                cmd += ['--wandb-project', args.wandb_project] + (['--wandb-entity', args.wandb_entity] if args.wandb_entity else [])
            print(f'[{time.strftime("%H:%M:%S")}] start {arm}/{env_id} on GPU {gpu} (attempt {attempts[key]})', flush=True)
            running[key] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE.parent)), gpu)
        time.sleep(args.poll_seconds)
    print('All tasks finished.', flush=True)
    if args.dashboard_port and not args.exit_when_done:
        print('Dashboard keeps serving; Ctrl-C to stop.', flush=True)
        while True:
            time.sleep(3600)


def start_dashboard(output, host, port):
    import popgym_bench_dashboard as dashboard
    thread = threading.Thread(target=dashboard.serve, args=(output, host, port), daemon=True)
    thread.start()
    print(f'Dashboard: http://{host}:{port}/  (all interfaces: open http://<b200-vpn-address>:{port}/)', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command')
    p.add_argument('--output', type=Path, default=Path(os.environ.get('POPGYM_BENCH_OUTPUT', 'runs/popgym-bench')))
    p.add_argument('--config', type=Path, help='JSON overrides of DEFAULTS (new study only)')
    p.add_argument('--envs', nargs='+', help='subset of task IDs (default: all 42)')
    p.add_argument('--gpus', nargs='+')
    p.add_argument('--workers-per-gpu', type=int, default=1, help='>1 measured slower in aggregate (compute-bound)')
    p.add_argument('--max-resumes', type=int, default=1)
    p.add_argument('--skip-control', action='store_true')
    p.add_argument('--skip-validation', action='store_true')
    p.add_argument('--poll-seconds', type=float, default=20.)
    p.add_argument('--wandb-project', default=os.environ.get('WANDB_PROJECT', STUDY))
    p.add_argument('--wandb-entity', default=os.environ.get('WANDB_ENTITY', 'fhstp-data-intelligence-research-group'))
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--dashboard-host', default='0.0.0.0', help='0.0.0.0 = reachable via the host VPN address')
    p.add_argument('--dashboard-port', type=int, default=8765, help='0 disables the dashboard')
    p.add_argument('--exit-when-done', action='store_true')
    w = sub.add_parser('worker')
    w.add_argument('--output', type=Path, required=True)
    w.add_argument('--arm', choices=tuple(ARMS), required=True)
    w.add_argument('--env', required=True)
    w.add_argument('--wandb-project')
    w.add_argument('--wandb-entity')
    v = sub.add_parser('validate')
    v.add_argument('--output', type=Path, required=True)
    d = sub.add_parser('dashboard')
    d.add_argument('--output', type=Path, required=True)
    d.add_argument('--host', default='0.0.0.0')
    d.add_argument('--port', type=int, default=8765)
    args = p.parse_args()
    if args.command == 'worker':
        train_task(args.output, args.arm, args.env, dict(project=args.wandb_project, entity=args.wandb_entity))
    elif args.command == 'validate':
        validate(args.output, VALIDATION_TASKS)
    elif args.command == 'dashboard':
        import popgym_bench_dashboard as dashboard
        dashboard.serve(args.output, args.host, args.port)
    else:
        if args.no_wandb:
            args.wandb_project = None
        supervise(args)


if __name__ == '__main__':
    main()
