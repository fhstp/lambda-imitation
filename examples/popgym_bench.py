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

STUDY = 'popgym-b200-flash-bench-v2'
DEFAULTS = dict(
    n_seeds=3, seed_base=42, max_steps=1_000_000, lambda_coef=10., fe_lr=1e-4, actor_lr=1e-4, critic_lr=1e-4,
    alpha_initial=1., alpha_final=.01, alpha_decay_steps=500_000, tau=.005, gamma=.99, lambda1=.05, lambda2=.99,
    memory_hidden_dim=512, projection_dim=128, actor_width=128, critic_width=256, online_buffer_size=100_000,
    prefill_steps=12_800, batch_transitions=8320, min_batch_episodes=8, collect_steps=16, chunk_steps=2048,
    eval_every=20_000, checkpoint_every=100_000, eval_episodes=50, final_episodes=200,
    noise_strength=1., noise_power=2., noise_max_duration=16,
    # Entropy handling (v2 study; LEGACY below = the original v1 study):
    alpha_scale='inverse_horizon',  # every alpha value above divided by the task horizon H ('none': unscaled)
    soft_main_target=True,       # main critic target adds +alpha*H(pi(.|s')) (SAC soft value); lambda heads stay reward-only
    autotune_alpha=True,         # SAC temperature learning towards target_entropy_fraction*log(A); schedule = initial value only
    target_entropy_fraction=.3, alpha_lr=1e-4)
# Values of options added after the v1 study, for saved configs that lack them.
LEGACY = dict(alpha_scale='none', soft_main_target=False, autotune_alpha=False)
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
    if c['alpha_scale'] not in ('none', 'inverse_horizon'):
        raise ValueError('alpha_scale must be none or inverse_horizon')
    if not 0 < c['target_entropy_fraction'] <= 1:
        raise ValueError('target_entropy_fraction must be in (0, 1]')
    return c


def saved_config(values):
    """Resolve a stored config; options it predates take their v1 values."""
    return resolve_config({**LEGACY, **{k: v for k, v in values.items() if k in DEFAULTS}})


def task_settings(c, horizon):
    """Constant transition budget per update: full episodes of length <= horizon.

    Battleship Easy (H=64) keeps the original 128 episodes x 65 slots."""
    return dict(batch_episodes=max(c['min_batch_episodes'], c['batch_transitions'] // (horizon + 1)))


def alpha_scale(c, horizon):
    return 1. / horizon if c.get('alpha_scale', 'none') == 'inverse_horizon' else 1.


def alpha_at(step, c, horizon=1):
    import jax.numpy as jnp
    t = jnp.clip(jnp.asarray(step) / c['alpha_decay_steps'], 0., 1.)
    a = jnp.exp((1 - t) * jnp.log(jnp.float32(c['alpha_initial'])) + t * jnp.log(jnp.float32(c['alpha_final'])))
    return a * jnp.float32(alpha_scale(c, horizon))


def agent_config(c, port, seed, small=False):
    s = task_settings(c, port.horizon)
    return dict(batch_size=s['batch_episodes'], online_buffer_size=c['online_buffer_size'], fe_lr=c['fe_lr'],
        actor_lr=c['actor_lr'], critic_lr=c['critic_lr'], alpha=c['alpha_initial'] * alpha_scale(c, port.horizon), gamma=c['gamma'], tau=c['tau'],
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
    sampler = flash.base.make_sampler(batch, port.horizon)
    if c.get('soft_main_target'):
        loss, components, context = soft_categorical_objective(fns, debug, sampler)
    else:  # the original study: unchanged production objective
        objective = clone(flash.categorical_objective, {'project_distribution': deterministic_projection, 'two_hot': two_hot})
        loss, components, context = objective(fns, debug, sampler, 'all-categorical')
    update_original = replace_captures(fns.update, {'loss_combined': loss})
    H = port.horizon

    if c.get('autotune_alpha'):
        import optax
        adam = optax.adam(c['alpha_lr'])
        target_entropy = c['target_entropy_fraction'] * float(jnp.log(port.action_dim))

        @jax.jit
        def update(state, key):
            # SAC temperature loss alpha*(H(pi) - target) on the batch entropy;
            # the schedule only provides the initial value (agent_config).
            state, metrics = update_original(state, key)
            grad = jnp.exp(state.log_alpha) * (metrics['entropy'] - target_entropy)
            step, opt = adam.update(grad, state.alpha_optimizer_state)
            log_alpha = optax.apply_updates(state.log_alpha, step)
            metrics = {**metrics, 'alpha': jnp.exp(log_alpha), 'target_entropy': jnp.float32(target_entropy)}
            return state._replace(log_alpha=log_alpha, alpha=jnp.exp(log_alpha), alpha_optimizer_state=opt), metrics
    else:
        @jax.jit
        def update(state, key):
            a = alpha_at(state.update_step, c, H)
            state, metrics = update_original(state._replace(alpha=a, log_alpha=jnp.log(a)), key)
            a = alpha_at(state.update_step, c, H)
            return state._replace(alpha=a, log_alpha=jnp.log(a)), metrics

    fns = fns._replace(update=update)
    _, bfns = create_buffer({k: v.shape[1:] for k, v in states[0].online_buffer.info.items()}, c['online_buffer_size'], 1, [], [])
    carry_dim = context['feature_extractor'].initialize_carry(1).shape[-1]
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    cls = HostRunner if getattr(port, 'is_host', False) else BenchRunner
    return stacked, cls(c, port, fns, debug, bfns, carry_dim), seeds


def soft_categorical_objective(fns, debug, sampler):
    """popgym_flash_prototypes.categorical_objective('all-categorical') with the
    deterministic projection, except the MAIN critic target: SAC soft value.

    The next-state mixture (min-twin-mean per action, target-policy weighted)
    is shifted by r + gamma*(1-d)*(z + alpha*H(pi(.|s'))). Exact in expectation
    (the critic uses means); per-action shifts -alpha*log pi(a') would need a
    projection per action (~12 GB/update at 144 actions). Lambda heads keep
    reward-only scalar Retrace two-hot targets. Actor loss as before."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from lambda_imitation.iqlearn import retrace_targets
    from popgym_gradient_balance import captures
    import popgym_flash_prototypes as flash
    from battleship_sweep import deterministic_projection
    support, SUPPORT, target_mixture = flash.support, flash.SUPPORT, flash.target_mixture
    two_hot = lambda values: deterministic_projection(jnp.ones((values.size, 1)), values.reshape(-1, 1))
    h = captures(captures(fns.update)['loss_combined']); hp = h['params']
    main = captures(h['loss_critic']); low = captures(h['loss_vtrace_lambda_sequence'])
    actor = captures(debug.get_entropy)['run_actor']
    graphs = (main['critic_graph'], h['lambda1_critic_graph'], h['lambda2_critic_graph'])
    flat = lambda x: x.reshape(-1, *x.shape[2:])
    def distributions(state, graph, x):
        return jnp.stack([nnx.merge(getattr(graph, branch), getattr(state, branch)).log_probs(x) for branch in ('q1', 'q2')], -1)
    def taken_log_probs(state, graph, x, actions):
        logp = distributions(state, graph, x)
        indices = jnp.round(actions.reshape(-1)).astype(jnp.int32)
        return logp[jnp.arange(len(indices)), indices]

    def components(fe, actor_state, critic, l1, l2, fe_target, actor_target, critic_target, l1_target, l2_target, alpha, raw, keys):
        valid = raw['episode_valid'].T[:-1].astype(jnp.float32)
        denominator = jnp.maximum(valid.sum(), 1.)
        mean = lambda v: (v.reshape(valid.shape) * valid).sum() / denominator
        x, y = debug.calculate_latent(fe, fe_target, raw['observations'], raw['actions'], raw['terminated'],
                                      jnp.zeros((hp.online_batch_size, h['feature_extractor'].initialize_carry(1).shape[-1])))
        act, reward, done, behaviour = [raw[k].swapaxes(0, 1) for k in ('actions', 'rewards', 'terminated', 'behaviour_weight')]
        actor_x = flat(jax.lax.stop_gradient(x[:-1]))
        probs = jax.nn.softmax(actor(actor_state, actor_x)); logp = jax.nn.log_softmax(actor(actor_state, actor_x))
        entropy = -jnp.sum(probs * logp, -1)
        q = captures(main['get_q_both'])['run_critic'](jax.lax.stop_gradient(critic), graphs[0], actor_x).min(-1)
        v = jnp.sum(probs * q, -1) + alpha * entropy
        actor_loss = -mean(v)
        next_logs = distributions(critic_target, graphs[0], flat(y[1:]))
        next_logits = actor(actor_target, flat(y[1:]))
        policy = jax.nn.softmax(next_logits)
        next_entropy = -jnp.sum(policy * jax.nn.log_softmax(next_logits), -1)
        mixture = target_mixture(next_logs, policy)
        bonus = alpha * next_entropy
        shifted = reward[:-1].reshape(-1, 1) + hp.gamma * (1. - done[:-1].reshape(-1, 1)) * (support()[None] + bonus[:, None])
        label = jax.lax.stop_gradient(deterministic_projection(mixture, shifted))
        predicted_logs = taken_log_probs(critic, graphs[0], flat(x[:-1]), flat(act[:-1]))
        critic_loss = mean(-jnp.sum(predicted_logs * label[..., None], -2).mean(-1))
        metrics = {'q': mean(jnp.sum(probs * q, -1)), 'entropy': mean(entropy), 'v': mean(v),
            'critic_loss': critic_loss, 'target_q': mean(jnp.sum(label * support(), -1)), 'alpha': alpha,
            'soft_bonus': mean(bonus), 'episode_mean_length': raw['episode_length'].mean(), 'valid_fraction': valid.mean(),
            'main_target_clipped_mass': mean(jnp.sum(mixture * ((shifted < SUPPORT[0]) | (shifted > SUPPORT[1])), -1)),
            'main_target_edge_mass': mean(label[:, 0] + label[:, -1])}
        losses = [actor_loss, critic_loss]
        for name, lam, online, target, graph, key in zip(('lambda1', 'lambda2'), (hp.lambda1, hp.lambda2),
                (l1, l2), (l1_target, l2_target), graphs[1:], keys[3:]):
            Tp, B = x.shape[:2]
            value = low['get_v'](actor_target, target, graph, jnp.float32(0), flat(y), key, False, False).reshape(Tp, B)
            ratios = low['get_importance_ratios'](actor_target, flat(y), flat(act), flat(behaviour)).reshape(Tp, B)
            taken = low['get_q'](target, graph, flat(y), flat(act)).reshape(Tp, B)
            returns = jax.lax.stop_gradient(retrace_targets(taken, value, reward, done, ratios, hp.gamma, lam, hp.c_bar))[:-1]
            label_l = jax.lax.stop_gradient(two_hot(returns.reshape(-1)))
            predicted = taken_log_probs(online, graph, flat(x[:-1]), flat(act[:-1]))
            loss = mean(-jnp.sum(predicted * label_l[..., None], -2).mean(-1))
            metrics[name + '_target_clipped_fraction'] = mean((returns < SUPPORT[0]) | (returns > SUPPORT[1]))
            metrics[name + '_target_edge_mass'] = mean(label_l[:, 0] + label_l[:, -1])
            losses.append(loss); metrics[name + '_loss'] = loss; metrics[name + '_target'] = mean(returns)
        ld1 = low['get_q'](jax.lax.stop_gradient(l1), graphs[1], flat(x[:-1]), flat(act[:-1]))
        ld2 = low['get_q'](jax.lax.stop_gradient(l2), graphs[2], flat(x[:-1]), flat(act[:-1]))
        discrepancy = mean((ld1 - ld2) ** 2)
        losses.append(hp.lambda_coef * discrepancy); metrics['ld_loss'] = discrepancy
        return sum(losses), metrics, jnp.stack(losses)

    def combined(fe, actor_state, critic, l1, l2, sf1, sf2, fe_target, actor_target, critic_target, l1_target, l2_target,
                 sf1_target, sf2_target, alpha, buffer, key):
        keys = jax.random.split(key, 5); sample, _ = sampler(buffer, keys[0])
        loss, metrics, _ = components(fe, actor_state, critic, l1, l2, fe_target, actor_target, critic_target,
                                      l1_target, l2_target, alpha, sample.this_info, keys)
        return loss, metrics
    return combined, components, h


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


EXIT_YIELDED, EXIT_PAUSED = 3, 4


class Yield(Exception):
    """Another attempt owns (or finished) this task; leave without uploading."""


def train_task(output, arm, env_id, wandb_cfg, small=False, deadline=None, upload_every=200_000, stop_margin=1200.):
    """One task (all seeds). Returns EXIT_YIELDED/EXIT_PAUSED/0.

    deadline (epoch seconds) or SIGTERM: checkpoint, upload, mark paused, exit.
    With W&B, the worker claims the task (popgym_bench_coord) and resumes
    from the newest checkpoint, local or W&B artifact."""
    import signal
    import socket
    import cloudpickle
    import jax
    import jax.numpy as jnp
    import numpy as np
    import popgym_bench_coord as coord
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
    stop = dict(requested=False)
    signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
    host = dict(host=socket.gethostname(), slurm_job=os.environ.get('SLURM_JOB_ID'))

    # Claim first (before minutes of compilation/prefill), so others see us early.
    run_id = hashlib.sha256(f'{STUDY}/{Path(output).resolve()}/{arm}/{env_id}'.encode()).hexdigest()[:16]
    run, api = None, None
    if wandb_cfg.get('project'):
        import wandb
        api = wandb.Api(timeout=60)
        run = wandb_safe(lambda: wandb.init(project=wandb_cfg['project'], entity=wandb_cfg.get('entity'), id=run_id,
            resume='allow', group=env_id, job_type=arm, name=f'{env_id.removeprefix("popgym-").removesuffix("-v0")}-{arm}',
            tags=[arm, meta['backend'], env_id], config={**c, **meta, **host, 'seeds': list(range(c['seed_base'], c['seed_base'] + c['n_seeds']))},
            dir=str(folder), settings=wandb.Settings(console='off')), folder)
        if run is not None:
            run.define_metric('steps')
            run.define_metric('*', step_metric='steps')
            # A resumed run id may still say 'yielded'/'paused' from before.
            wandb_safe(lambda: run.summary.update({'run_state': 'starting'}), folder)
    log = (lambda payload: wandb_safe(lambda: run.log(payload), folder)) if run is not None else (lambda payload: None)
    entity, project = wandb_cfg.get('entity'), wandb_cfg.get('project')

    def check_claim():
        if run is None:
            return
        tasks = wandb_safe(lambda: coord.board(coord.fetch_records(api, entity, project)), folder)
        if tasks is None:
            return  # W&B unreachable: keep training rather than abandon work
        task = tasks.get((arm, env_id))
        if task and task['complete']:
            raise Yield('task already complete')
        holder = coord.owner(task)
        if holder is not None and holder != run.id:
            raise Yield(f'task owned by attempt {holder}')

    try:
        check_claim()
        # Newest checkpoint wins: local file or W&B artifact (another host/job).
        local_steps = None
        if checkpoint.exists():
            with checkpoint.open('rb') as f:
                local_steps = cloudpickle.load(f)['steps']
        if api is not None:
            artifact, remote_steps = wandb_safe(lambda: coord.latest_checkpoint(api, entity, project, arm, env_id), folder) or (None, 0)
            if artifact is not None and (local_steps is None or remote_steps > local_steps):
                artifact.download(root=str(folder / 'download'))
                os.replace(folder / 'download' / 'latest.pkl', checkpoint)
                print(f'Resuming {arm}/{env_id} from W&B checkpoint at {remote_steps} steps', flush=True)
    except Yield as reason:
        return yield_task(run, api, folder, str(reason))
    saved = None
    if checkpoint.exists():
        with checkpoint.open('rb') as f:
            saved = cloudpickle.load(f)
        if saved_config(saved['config']) != c or saved['meta'] != meta:
            raise ValueError('Checkpoint configuration differs; use a new output directory')
        if saved['source_manifest'] != manifest:
            # Resuming with changed sources is recorded, not silently hidden.
            with (folder / 'resume-source-changes.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=time.time(), old=saved['source_manifest'], new=manifest)) + '\n')
    atomic_json(folder / 'config.json', dict(config=c, meta=meta, source_manifest=manifest, versions=runtime_versions(),
                                             devices=[str(d) for d in jax.devices()], **host))
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
    uploaded = dict(steps=saved['steps'] if saved else -1)

    def checkpoint_now():
        save_checkpoint(checkpoint, dict(state=jax.device_get(states), rollout=host_rollout(rollout), config=c, meta=meta,
            steps=step, history=history, interactions=interactions, source_manifest=manifest, wandb_id=run_id))

    def upload():
        if run is None or uploaded['steps'] == step:
            return
        done = wandb_safe(lambda: coord.upload_checkpoint(run, api, entity, project, checkpoint, arm, env_id, step), folder)
        if done is not None:
            uploaded['steps'] = done
            wandb_safe(lambda: run.summary.update({'checkpoint_steps': done}), folder)

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

    def log_eval(row, prefix='eval'):
        payload = {'steps': row['steps'], 'run_state': 'running'}
        if 'steps_per_second' in row:
            payload['throughput/steps_per_second'] = row['steps_per_second']
        for mode, metrics_ in row['eval'].items():
            for k, s in metrics_.items():
                payload[f'{prefix}/{mode}/{k}/mean'] = s['mean']; payload[f'{prefix}/{mode}/{k}/std'] = s['std']
            # Per-seed returns: the remote dashboard needs seed bands/best-during-training.
            for seed, value in zip(seeds, metrics_['return_mean']['seeds']):
                payload[f'seed/{seed}/{prefix}/{mode}/return_mean'] = value
        for k, s in row.get('train', {}).items():
            payload[f'train/{k}/mean'] = s['mean']
        log(payload)

    def pause(reason):
        checkpoint_now()
        upload()
        atomic_json(folder / 'status.json', dict(state='paused', steps=step, target_steps=c['max_steps'], time=time.time(), reason=reason))
        if run is not None:
            wandb_safe(lambda: run.summary.update({'run_state': 'paused', 'steps': step, 'pause_reason': reason}), folder)
        print(f'Paused {arm}/{env_id} at {step} steps ({reason})', flush=True)
        return EXIT_PAUSED

    started, start_step, chunk_seconds = time.monotonic(), step, 0.
    try:
        check_claim()  # definitive: everyone started at the same time is visible by now
        if not history:
            history.append(evaluate(False))
            log_eval(history[-1])
            atomic_json(folder / 'history.json', history)
            checkpoint_now()
        next_eval = (step // c['eval_every'] + 1) * c['eval_every']
        while step < c['max_steps']:
            if stop['requested']:
                return pause('SIGTERM')
            if deadline is not None and time.time() + stop_margin + 3 * chunk_seconds > deadline:
                return pause('time limit')
            n = min(c['chunk_steps'], next_eval - step)
            tick = time.monotonic()
            states, rollout, metrics = runner.train(states, rollout, n)
            metrics = jax.device_get(metrics)
            chunk_seconds = max(chunk_seconds, time.monotonic() - tick) if chunk_seconds else time.monotonic() - tick
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
            log_eval(row)
            atomic_json(folder / 'history.json', history)
            if step % c['checkpoint_every'] == 0 or step == c['max_steps']:
                checkpoint_now()
                if upload_every and step % upload_every == 0 and step < c['max_steps']:
                    upload()
            check_claim()
        final = evaluate(True)
        history.append(final)
        atomic_json(folder / 'history.json', history)
        log_eval(final, 'final')
        if run is not None:
            wandb_safe(lambda: run.summary.update({'final_stochastic_return': final['eval']['stochastic']['return_mean']['mean'],
                                                   'final_greedy_return': final['eval']['greedy']['return_mean']['mean'],
                                                   'run_state': 'complete'}), folder)
        atomic_json(folder / 'status.json', dict(state='complete', steps=step, target_steps=c['max_steps'], time=time.time()))
        return 0
    except Yield as reason:
        return yield_task(run, api, folder, str(reason))
    except BaseException:
        error = traceback.format_exc()[-4000:]
        atomic_json(folder / 'status.json', dict(state='failed', steps=step, target_steps=c['max_steps'], time=time.time(), error=error))
        if run is not None:
            wandb_safe(lambda: run.summary.update({'run_state': 'failed', 'error': error[-1500:]}), folder)
            wandb_safe(lambda: run.finish(exit_code=1), folder)
            run = None
        raise
    finally:
        if run is not None:
            wandb_safe(lambda: run.finish(), folder)


def yield_task(run, api, folder, reason):
    """Leave a task to its owner. The run is marked, not deleted: W&B never
    re-accepts a deleted run id, and ids are deterministic per output path."""
    atomic_json(folder / 'status.json', dict(state='yielded', time=time.time(), reason=reason))
    print(f'Yielding: {reason}', flush=True)
    if run is not None:
        wandb_safe(lambda: run.summary.update({'run_state': 'yielded', 'yield_reason': reason}), folder)
        wandb_safe(lambda: run.finish(), folder)
    return EXIT_YIELDED


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
    c = saved_config(json.loads((Path(output) / 'study.json').read_text())['config'])
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


class Adopted:
    """A worker of this output started by an earlier supervisor (still alive)."""

    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        try:
            os.kill(self.pid, 0)
            return None
        except ProcessLookupError:
            self.returncode = -1
            return self.returncode

    def send_signal(self, sig):
        try:
            os.kill(self.pid, sig)
        except ProcessLookupError:
            pass


def find_workers(output):
    """Live `popgym_bench.py worker --output OUTPUT` processes, via /proc."""
    found = {}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        argv = [a.decode(errors='replace') for a in argv if a]
        if 'worker' not in argv or not any(a.endswith('popgym_bench.py') for a in argv):
            continue
        get = lambda flag: argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None
        if get('--output') and Path(get('--output')).resolve() == Path(output).resolve() and int(proc.name) != os.getpid():
            gpu = None
            try:
                env = (proc / 'environ').read_bytes().split(b'\0')
                gpu = next((e.split(b'=', 1)[1].decode() for e in env if e.startswith(b'CUDA_VISIBLE_DEVICES=')), None)
            except OSError:
                pass
            found[(get('--arm'), get('--env'))] = (Adopted(int(proc.name)), gpu)
    return found


def study_from_wandb(entity, project):
    """The study config/task list recorded by the first supervisor in W&B."""
    import wandb
    runs = [r for r in wandb.Api(timeout=60).runs(f'{entity}/{project}') if r.job_type == 'supervisor' and r.config.get('study')]
    if not runs:
        raise RuntimeError(f'No supervisor run with a study config in {entity}/{project}; start the study once with a config first')
    first = min(runs, key=lambda r: r.created_at)
    return dict(first.config['study']), list(first.config['envs'])


def supervise(args):
    import signal
    # Workers get the original environment; the supervisor itself must never
    # initialize (and preallocate) a GPU while reading task metadata.
    worker_env = dict(os.environ)
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import popgym_bench_coord as coord
    from popgym_bench_envs import all_env_ids
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    study = output / 'study.json'
    if args.study_from_wandb and not study.exists():
        config, envs = study_from_wandb(args.wandb_entity, args.wandb_project)
        config = saved_config(config)
        envs = args.envs or envs
    else:
        config = resolve_config(json.loads(args.config.read_text()) if args.config else {})
        envs = args.envs or all_env_ids()
    record = dict(study=STUDY, config=config, arms=ARMS, wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
                  envs=envs, created=time.time())
    if study.exists():
        old = json.loads(study.read_text())
        if saved_config(old['config']) != config and not args.study_from_wandb or (args.envs and old['envs'] != record['envs']):
            raise ValueError(f'{study} has a different configuration/task list; use a new --output')
        record = old
    else:
        atomic_json(study, record)
    order = task_order(record['envs'])
    if args.order == 'cheap-first':
        order = order[::-1]
    gpus = args.gpus or detect_gpus()
    if not gpus:
        raise RuntimeError('No GPUs detected; pass --gpus')
    slots = [g for g in gpus for _ in range(args.workers_per_gpu)]
    deadline = time.time() + args.time_limit * 3600 if args.time_limit else None
    print(f'{len(order)} tasks x {len(ARMS)} arms on GPUs {gpus} ({args.workers_per_gpu} workers each)'
          + (f', at most {args.max_tasks} tasks' if args.max_tasks else '')
          + (f', stop by {time.strftime("%Y-%m-%d %H:%M", time.localtime(deadline))}' if deadline else ''), flush=True)
    if args.dashboard_port:
        start_dashboard(output, args.dashboard_host, args.dashboard_port)
    running = find_workers(output) if args.adopt else {}
    for (arm, env_id), (_, gpu) in running.items():
        print(f'[{time.strftime("%H:%M:%S")}] adopted running {arm}/{env_id} on GPU {gpu}', flush=True)
    proof = output / 'validation.json'
    current = source_manifest()
    if not args.skip_validation and not (proof.exists() and json.loads(proof.read_text()).get('source_manifest') == current):
        gate_gpu = next((g for g in gpus if g not in [gpu for _, gpu in running.values()]), gpus[0])
        print(f'Validation gate on GPU {gate_gpu}: {", ".join(VALIDATION_TASKS)}', flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), 'validate', '--output', str(output)]
        env = {**worker_env, 'CUDA_VISIBLE_DEVICES': gate_gpu, 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'}
        with open(output / 'validation.log', 'a') as log:
            code = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE.parent))
        if code:
            raise SystemExit(f'Validation gate FAILED (see {output / "validation.log"}); nothing launched')
        print('Validation passed.', flush=True)
    supervisor_run = start_supervisor_run(args, output, record, len(slots))
    api = None
    if args.wandb_project:
        import wandb
        api = wandb.Api(timeout=60)
    published, board_at, tasks_board = 0., 0., {}
    attempts, finished, launched = {}, set(), 0
    stopping = dict(flag=False)

    def on_term(*_):
        if not stopping['flag']:
            print(f'[{time.strftime("%H:%M:%S")}] SIGTERM: asking workers to checkpoint and pause', flush=True)
        stopping['flag'] = True
        for proc, _ in running.values():
            proc.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, on_term)
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    while True:
        for key, (proc, slot) in list(running.items()):
            if proc.poll() is not None:
                del running[key]
                arm, env_id = key
                state = read_status(task_folder(output, arm, env_id)).get('state')
                print(f'[{time.strftime("%H:%M:%S")}] {arm}/{env_id} exited {proc.returncode} ({state})', flush=True)
                if proc.returncode == EXIT_YIELDED or state == 'yielded':
                    launched -= 1  # did not use up one of this job's task slots
                    finished.add(key)
                elif proc.returncode == EXIT_PAUSED or state in ('paused', 'complete'):
                    finished.add(key)
        if api is not None and time.time() - board_at > 60:
            board = wandb_safe(lambda: coord.board(coord.fetch_records(api, args.wandb_entity, args.wandb_project)), output)
            if board is not None:
                tasks_board, board_at = board, time.time()
        local_ids = {hashlib.sha256(f'{STUDY}/{output}/{arm}/{env_id}'.encode()).hexdigest()[:16] for arm, env_id in running}

        def pending(arm):
            result = []
            for env_id in order:
                key = (arm, env_id)
                if key in running or key in finished:
                    continue
                if read_status(task_folder(output, arm, env_id)).get('state') == 'complete':
                    continue
                # Done or claimed on another host/job (W&B board, refreshed every minute).
                if not coord.available(tasks_board, key, local_ids):
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
        if args.max_tasks and queue:
            # Nodes started together would all pick the same cheapest tasks and
            # then yield to each other; spread them over the head of the queue.
            import random
            head = queue[:3 * len(slots)]
            random.Random(f'{socket_name()}-{os.environ.get("SLURM_JOB_ID")}-{os.getpid()}').shuffle(head)
            queue = head + queue[3 * len(slots):]
        if stopping['flag'] or (args.max_tasks and launched >= args.max_tasks) or (deadline and time.time() > deadline - min(3600, args.time_limit * 900)):
            queue = []  # no new tasks: stopping, quota used, or too little time left (1 h, or 1/4 of short limits)
        atomic_json(output / 'scheduler.json', dict(time=time.time(), gpus=gpus, slots=len(slots),
            phase='primary' if primary_busy else 'control', queued=[list(k) for k in queue],
            running=[dict(arm=k[0], env_id=k[1], gpu=slot) for k, (_, slot) in running.items()],
            attempts=[dict(arm=k[0], env_id=k[1], attempts=v) for k, v in attempts.items()]))
        if supervisor_run is not None and time.time() - published > 60:
            published = time.time()
            wandb_safe(lambda: supervisor_run.summary.update({'scheduler': (output / 'scheduler.json').read_text(),
                                                              'heartbeat': published}), output)
        if not queue and not running:
            break
        free = list(slots)
        for _, slot in running.values():
            if slot in free:
                free.remove(slot)
        for key in queue:
            if not free or (args.max_tasks and launched >= args.max_tasks):
                break
            gpu = free.pop(0)
            arm, env_id = key
            attempts[key] = attempts.get(key, 0) + 1
            launched += 1
            log = (logs / f'{arm}-{env_id}.log').open('a')
            env = {**worker_env, 'CUDA_VISIBLE_DEVICES': gpu, 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
                   'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
            cmd = [sys.executable, str(Path(__file__).resolve()), 'worker', '--output', str(output), '--arm', arm, '--env', env_id,
                   '--upload-every', str(args.upload_every), '--stop-margin', str(args.stop_margin)]
            if deadline:
                cmd += ['--deadline', str(deadline)]
            if args.wandb_project:
                cmd += ['--wandb-project', args.wandb_project] + (['--wandb-entity', args.wandb_entity] if args.wandb_entity else [])
            print(f'[{time.strftime("%H:%M:%S")}] start {arm}/{env_id} on GPU {gpu} (attempt {attempts[key]})', flush=True)
            running[key] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE.parent)), gpu)
        time.sleep(args.poll_seconds)
    print('All tasks finished.' if not (args.max_tasks or deadline or stopping['flag']) else 'Nothing left for this job; exiting.', flush=True)
    if supervisor_run is not None:
        wandb_safe(lambda: supervisor_run.summary.update({'scheduler': (output / 'scheduler.json').read_text(), 'all_finished': True}), output)
        wandb_safe(lambda: supervisor_run.finish(), output)
    if args.dashboard_port and not args.exit_when_done:
        print('Dashboard keeps serving; Ctrl-C to stop.', flush=True)
        while True:
            time.sleep(3600)


def upload_checkpoints(args):
    """Push every unfinished task's local checkpoint to W&B (e.g. before a host
    goes away), so any other host can resume it. Running workers are untouched."""
    import cloudpickle
    import wandb
    import popgym_bench_coord as coord
    output = Path(args.output).resolve()
    api = wandb.Api(timeout=60)
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, job_type='checkpoint-upload',
                     name=f'checkpoint-upload-{socket_name()}', dir=str(output), settings=wandb.Settings(console='off'))
    for folder in sorted(output.glob('*/popgym-*')):
        arm, env_id = folder.parent.name, folder.name
        path = folder / 'latest.pkl'
        if arm not in ARMS or not path.exists() or read_status(folder).get('state') == 'complete':
            continue
        with path.open('rb') as f:
            steps = cloudpickle.load(f)['steps']
        _, remote = coord.latest_checkpoint(api, args.wandb_entity, args.wandb_project, arm, env_id)
        if remote >= steps:
            print(f'{arm}/{env_id}: W&B already has {remote} >= {steps} steps', flush=True)
            continue
        coord.upload_checkpoint(run, api, args.wandb_entity, args.wandb_project, path, arm, env_id, steps)
        print(f'{arm}/{env_id}: uploaded checkpoint at {steps} steps', flush=True)
    run.finish()


def socket_name():
    import socket
    return socket.gethostname()


def start_supervisor_run(args, output, record, slots):
    """Tiny W&B run carrying study config, launch gate and scheduler state,
    so a dashboard on another machine can show them (popgym_bench_wandb.py)."""
    if not args.wandb_project:
        return None
    import wandb
    proof = output / 'validation.json'
    validation = json.loads(proof.read_text()) if proof.exists() else None
    job = os.environ.get('SLURM_JOB_ID')
    run = wandb_safe(lambda: wandb.init(project=args.wandb_project, entity=args.wandb_entity, job_type='supervisor',
        group='supervisor', name='supervisor' if not job else f'node-{socket_name()}-{job}', resume='allow', dir=str(output),
        id=hashlib.sha256(f'{STUDY}/{output}/supervisor'.encode()).hexdigest()[:16],
        config=dict(study=record['config'], envs=record['envs'], host=socket_name(), slurm_job=job, slots=slots),
        settings=wandb.Settings(console='off')), output)
    if run is not None and validation is not None:
        wandb_safe(lambda: run.summary.update({'validation': json.dumps({k: v for k, v in validation.items() if k != 'source_manifest'})}), output)
    return run


def start_dashboard(output, host, port):
    import popgym_bench_dashboard as dashboard
    thread = threading.Thread(target=dashboard.serve, args=(output, host, port), daemon=True)
    thread.start()
    print(f'Dashboard: http://{host}:{port}/  (all interfaces: open http://<b200-vpn-address>:{port}/)', flush=True)


def add_supervisor_args(p, node=False):
    p.add_argument('--output', type=Path, default=Path(os.environ.get('POPGYM_BENCH_OUTPUT', default_node_output() if node else 'runs/popgym-bench')))
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
    p.add_argument('--dashboard-port', type=int, default=0 if node else 8765, help='0 disables the dashboard')
    p.add_argument('--exit-when-done', action='store_true', default=node)
    p.add_argument('--max-tasks', type=int, default=None, help='launch at most N tasks in this invocation (node default: one per GPU)')
    p.add_argument('--time-limit', type=float, default=None, help='hours; workers checkpoint, upload and pause before it runs out')
    p.add_argument('--order', choices=('lpt', 'cheap-first'), default='cheap-first' if node else 'lpt')
    p.add_argument('--no-adopt', dest='adopt', action='store_false', help='do not adopt live workers of this --output')
    p.add_argument('--study-from-wandb', action='store_true', default=node, help='take config/task list from the W&B study')
    p.add_argument('--upload-every', type=int, default=200_000, help='steps between W&B checkpoint uploads (0: only on pause)')
    p.add_argument('--stop-margin', type=float, default=1200., help='seconds before the time limit to checkpoint/upload/pause')


def slurm_hours_left():
    """Remaining wall time of this SLURM job in hours (None outside SLURM)."""
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        return None
    try:
        left = subprocess.check_output(['squeue', '-h', '-j', job, '-o', '%L'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return parse_slurm_time(left)


def parse_slurm_time(text):
    """'1-02:03:04', '02:03:04', '03:04' -> hours; 'UNLIMITED'/'' -> None."""
    if not text or not text[0].isdigit():
        return None
    days, _, rest = text.rpartition('-')
    parts = [int(x) for x in rest.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days or 0) * 24 + parts[0] + parts[1] / 60 + parts[2] / 3600


def default_node_output():
    import socket
    base = os.environ.get('TMPDIR') or '/tmp'
    return str(Path(base) / f'popgym-bench-{socket.gethostname()}-{os.environ.get("SLURM_JOB_ID", os.getpid())}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command')
    add_supervisor_args(p)
    n = sub.add_parser('node', help='cluster node: claim up to one task per GPU via W&B, train, exit')
    add_supervisor_args(n, node=True)
    w = sub.add_parser('worker')
    w.add_argument('--output', type=Path, required=True)
    w.add_argument('--arm', choices=tuple(ARMS), required=True)
    w.add_argument('--env', required=True)
    w.add_argument('--wandb-project')
    w.add_argument('--wandb-entity')
    w.add_argument('--deadline', type=float)
    w.add_argument('--upload-every', type=int, default=200_000)
    w.add_argument('--stop-margin', type=float, default=1200.)
    v = sub.add_parser('validate')
    v.add_argument('--output', type=Path, required=True)
    u = sub.add_parser('upload-checkpoints', help='push unfinished local checkpoints to W&B')
    u.add_argument('--output', type=Path, required=True)
    u.add_argument('--wandb-project', default=os.environ.get('WANDB_PROJECT', STUDY))
    u.add_argument('--wandb-entity', default=os.environ.get('WANDB_ENTITY', 'fhstp-data-intelligence-research-group'))
    d = sub.add_parser('dashboard')
    d.add_argument('--output', type=Path, required=True)
    d.add_argument('--host', default='0.0.0.0')
    d.add_argument('--port', type=int, default=8765)
    args = p.parse_args()
    if args.command == 'worker':
        sys.exit(train_task(args.output, args.arm, args.env, dict(project=args.wandb_project, entity=args.wandb_entity),
                            deadline=args.deadline, upload_every=args.upload_every, stop_margin=args.stop_margin))
    elif args.command == 'validate':
        validate(args.output, VALIDATION_TASKS)
    elif args.command == 'upload-checkpoints':
        upload_checkpoints(args)
    elif args.command == 'dashboard':
        import popgym_bench_dashboard as dashboard
        dashboard.serve(args.output, args.host, args.port)
    else:
        if args.no_wandb:
            args.wandb_project = None
        if args.command == 'node':
            if not args.wandb_project:
                raise SystemExit('node mode coordinates through W&B')
            if args.time_limit is None:
                args.time_limit = slurm_hours_left()
            if args.max_tasks is None:
                args.max_tasks = len(args.gpus or detect_gpus())
        supervise(args)


if __name__ == '__main__':
    main()
