"""RockSample on the shared runner, with a probe for what the memory retains.

RockSample is the λ-discrepancy paper's other grid POMDP, and structurally it is
the one that should suit an off-policy memory method best:

* The agent's **position is fully observed** and each rock has its **own slot**
  in the observation, so there is no aliasing to untangle from the action stream
  — which is why the reference sets ``action_concat=False`` here, unlike
  Battleship and PocMan.
* What is hidden is each rock's **goodness**, read through a noisy sensor whose
  accuracy falls with distance (``half_efficiency_distance``).  A check
  *overwrites* that rock's slot rather than accumulating, so combining repeated
  readings needs memory.
* ``sampled_rocks`` is **not** in the observation at all, and sampling clears the
  rock's slot to 0 — indistinguishable from "never checked".  Re-sampling a rock
  you already took pays the bad-rock reward.  That is a long-horizon memory
  requirement with a ±10 value gap attached, which is what the λ-discrepancy
  needs in order to have anything to measure.

The probe therefore has two heads, exactly like Battleship's: **goodness** (the
hidden state) and **sampled** (what the memory should know about its own
history), split into checked rocks = RETENTION and unchecked = INFERENCE.

Sizes follow the paper: ``rocksample_11_11`` (11 rocks, γ=0.99, H=128) and
``rocksample_15_15`` (15 rocks, γ=0.999, H=256).  The smaller 5_5 / 7_8 configs
are kept as controls: the memory horizon grows with the grid, and our T-maze
measurement says the discrepancy that survives off-policy data falls as
``keep**(2H)``, so the size sweep is the experiment, not just a difficulty knob.

Usage:
    python rocksample_probe.py                              # 11x11, paper arch
    python rocksample_probe.py --config rocksample_15_15 --num-seeds 5 --wandb
    python rocksample_probe.py --config rocksample_5_5     # short-horizon control
"""

import argparse
import json
import os
import pickle
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_common as common

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="RockSample probe.")

g = parser.add_argument_group("environment")
g.add_argument("--config", default="rocksample_11_11",
               choices=("rocksample_5_5", "rocksample_7_8",
                        "rocksample_11_11", "rocksample_15_15"),
               help="which RockSample config (the paper runs 11_11 and 15_15; "
                    "the smaller two are short-horizon controls)")
g.add_argument("--env-seed", type=int, default=0,
               help="PRNG seed for the rock LAYOUT, which is drawn once at "
                    "construction.  Kept separate from --seed so every agent "
                    "seed faces the same map (comparability); vary it to "
                    "resample the map.")
g.add_argument("--eval-max-steps", type=int, default=300,
               help="episode cap for evaluation scans (the env's own limit is "
                    "1000; episodes end early by exiting east)")

g = parser.add_argument_group("network architecture")
g.add_argument("--paper-arch", dest="paper_arch", action="store_true",
               help="reproduce the reference DiscreteActorCriticRNN: "
                    "Dense(H)->ReLU embedding, GRU(H), single-hidden actor and "
                    "critic heads of width H, no critic LayerNorm.  H defaults "
                    "to the paper's per-config value (128 for 11x11, 256 for "
                    "15x15) unless --memory-hidden-dim is given.  Default on.")
g.add_argument("--no-paper-arch", dest="paper_arch", action="store_false")
parser.set_defaults(paper_arch=True)
g.add_argument("--use-prev-action", dest="use_prev_action", action="store_true",
               help="feed the previous action to the embedding.  OFF by default, "
                    "matching the reference (action_concat is not set for "
                    "RockSample): each rock already has its own observation slot, "
                    "so nothing needs the action stream to be disambiguated.")
parser.set_defaults(use_prev_action=False)
g.add_argument("--lambda1", type=float, default=0.1,
               help="λ of the first λ-critic (default 0.1, the reference's "
                    "lambda0 for both RockSample LD configs)")
g.add_argument("--lambda2", type=float, default=0.95,
               help="λ of the second λ-critic (default 0.95; the reference uses "
                    "0.95 on 11x11 and 0.5 on 15x15)")

common.add_common_args(
    parser, output_dir_default="./rocksample_output",
    wandb_project_default="offline-lambda-rocksample-results")

# The reference's RockSample settings, where they carry over to an off-policy
# learner: heavy exploration (entropy_coeff 0.35 there), and the LD weight it
# selected.  The learning rates are ours -- theirs are PPO's single lr.
parser.set_defaults(
    rounds=30, train_steps=10_000, alpha=0.35, autotune_alpha=False,
    target_entropy=0.0, tau=0.005, memory_type="gru",
    fe_lr=1e-4, actor_lr=1e-4, critic_lr=1e-4,
    batch_size=128, sequence_length=40, burn_in_length=10,
    online_buffer_size=200_000, lambda_coef=0.5, retrace=True, use_sac=False,
    memory_hidden_dim=0,        # 0 = take the paper's value for --config
)

args = common.parse_args(parser, extra_flags_env="ROCKSAMPLE_EXTRA_FLAGS")
_SWEEP_RUN = common.apply_sweep_config(parser, args)

# ── always-needed imports ────────────────────────────────────────────────────

import jax
import jax.numpy as jnp
import numpy as np

# ── config ───────────────────────────────────────────────────────────────────

try:
    import lambda_envs
    from lambda_envs.envs.rocksample import RockSample
except ImportError:
    sys.exit("lambda-envs required.  pip install lambda-envs")

CONFIG_PATH = (os.path.join(os.path.dirname(lambda_envs.__file__), "configs",
                            f"{args.config}_config.json"))
with open(CONFIG_PATH) as f:
    ENV_CONFIG = json.load(f)

# γ lives in the config (0.99 for 11x11, 0.999 for 15x15) and the env class does
# not expose it, so take it from there unless the caller overrode --gamma.
_GAMMA_DEFAULT = parser.get_default("gamma")
if args.gamma == _GAMMA_DEFAULT and "gamma" in ENV_CONFIG:
    args.gamma = float(ENV_CONFIG["gamma"])

# the paper's hidden size per config: 128 for 11x11 (BatchHyperparams default),
# 256 for 15x15 (set explicitly in its hyperparameter files)
if args.memory_hidden_dim == 0:
    args.memory_hidden_dim = 256 if args.config == "rocksample_15_15" else 128

_wandb = common.init_wandb(args, {
    "env": "RockSample",
    "experiment": "rock_probe",
    "config": args.config,
    "size": ENV_CONFIG["size"], "rocks": ENV_CONFIG["rocks"],
    "half_efficiency_distance": ENV_CONFIG["half_efficiency_distance"],
    "exit_reward": ENV_CONFIG["exit_reward"],
    "env_seed": args.env_seed,
    "paper_arch": args.paper_arch, "use_prev_action": args.use_prev_action,
    "lambda1": args.lambda1, "lambda2": args.lambda2,
    "algo": ("SAC+LD" if args.approximate_lambda else "SAC"),
    **common.common_wandb_config(args),
}, _SWEEP_RUN)

init_probe_params = common.init_probe_params
probe_forward = common.probe_forward

os.makedirs(args.output_dir, exist_ok=True)

# ── env ──────────────────────────────────────────────────────────────────────

import optax
from tqdm.rich import tqdm

from lambda_imitation.iqlearn import Hyperparameters
from lambda_imitation.utils import (create_iqlearn_from_env, env_spec_from_gymnax,
                                    relu_projection)


class _Adapter:
    """RockSample.get_obs takes no params; gymnax callers pass one."""

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def get_obs(self, state, params=None):
        return self._wrapped.get_obs(state)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


_raw_env = RockSample(jax.random.key(args.env_seed), config_path=CONFIG_PATH)
env = _Adapter(_raw_env)
env_params = env.default_params
spec = env_spec_from_gymnax(env, env_params)
K = int(_raw_env.k)                     # rocks
SIZE = int(_raw_env.size)
NUM_ACTIONS = int(spec.action_dim)      # K + 5

print(f"RockSample {args.config}: {SIZE}x{SIZE}, {K} rocks, "
      f"obs {spec.obs_shape}, {NUM_ACTIONS} actions, gamma {args.gamma}, "
      f"exit reward {ENV_CONFIG['exit_reward']} (walking east ends the episode, "
      f"so that is the do-nothing floor)")

CARRY_DIM = (0 if args.memory_type == "identity"
             else 2 * args.memory_hidden_dim if args.memory_type == "lstm"
             else args.memory_hidden_dim)
if CARRY_DIM == 0:
    sys.exit("The probe needs a recurrent memory (--memory-type rnn/gru/lstm).")

if args.paper_arch:
    H = args.memory_hidden_dim
    projection_arg = relu_projection(H)
    actor_dims = critic_dims = (H,)
    critic_layer_norm = False
else:
    projection_arg = args.projection_dim if args.projection_dim > 0 else None
    actor_dims = critic_dims = (256, 256)
    critic_layer_norm = args.critic_layer_norm


def zero_carry():
    return jnp.zeros((CARRY_DIM,), dtype=jnp.float32)


def zero_prev_action():
    return jnp.zeros((NUM_ACTIONS if args.use_prev_action else 0,), dtype=jnp.float32)


hp = Hyperparameters(
    batch_size=args.batch_size, online_buffer_size=args.online_buffer_size,
    target_entropy=args.target_entropy,
    fe_lr=args.fe_lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
    lambda_critic_lr=args.critic_lr, alpha_lr=1e-4,
    alpha=args.alpha, autotune_alpha=args.autotune_alpha,
    gamma=args.gamma, tau=args.tau,
    lambda1=args.lambda1, lambda2=args.lambda2,
    c_bar=1.0, rho_bar=1.0, lambda_truncation=20,
    sequence_length=args.sequence_length, burn_in_length=args.burn_in_length,
    lambda_coef=args.lambda_coef, fake_onpolicy_loss=False,
    actor_critic=args.actor_critic,
    gvd_coef=args.gvd_coef, gvd_lambda1=args.gvd_lambda1,
    gvd_lambda2=args.gvd_lambda2, gvd_sf_lr=args.gvd_sf_lr,
    gvd_stop_fe=args.gvd_stop_fe,
    stop_actor_fe=args.stop_actor_fe, stop_critic_fe=args.stop_critic_fe,
    ld_center=args.ld_center, retrace=args.retrace,
    per_alpha=args.per_alpha, per_beta=args.per_beta,
    per_ratio_floor=args.per_ratio_floor, per_window=args.per_window,
)

_MAX_STEPS = min(int(env_params.max_steps_in_episode), args.eval_max_steps)
expert_data = {"observations": jnp.zeros((1, *spec.obs_shape), dtype=jnp.float32),
               "actions": jnp.zeros((1, 1), dtype=jnp.float32)}


def _build_agent(seed_val):
    return create_iqlearn_from_env(
        spec, expert_data, buffer_size=1, hp=hp, projection=projection_arg,
        memory_type=args.memory_type, memory_hidden_dim=args.memory_hidden_dim,
        actor_dims=actor_dims, critic_dims=critic_dims,
        lambda1_critic_dims=critic_dims, lambda2_critic_dims=critic_dims,
        train_steps=args.train_steps, approximate_lambda=args.approximate_lambda,
        use_prev_action=args.use_prev_action, critic_layer_norm=critic_layer_norm,
        burn_in_from_stored_carry=args.burn_in_from_stored_carry,
        use_gvd=args.gvd, gvd_sf_dims=(256, 256),
        debug=True, seed=seed_val, use_sac=args.use_sac)


tag = ("SAC+LD" if args.approximate_lambda else "SAC") + ("+GVD" if args.gvd else "")
print(f"Building {tag} agent (memory={args.memory_type}, "
      f"hidden={args.memory_hidden_dim}, paper_arch={args.paper_arch})…")
state, fns, debug_fns = _build_agent(args.seed)

_transplant_fe, _save_fe = common.make_fe_checkpointer(args.init_fe, args.output_dir)
_eval_kwargs = dict(zero_carry=zero_carry, zero_prev_action=zero_prev_action,
                    max_steps=_MAX_STEPS)
evaluate = common.make_evaluate(fns, env, env_params, **_eval_kwargs)

# ── rollout collection ───────────────────────────────────────────────────────


@partial(jax.jit, static_argnames=["n_steps"])
def collect_rollout(agent_state, key, n_steps):
    """Record (carry, rock goodness, checked mask, sampled mask) per step."""
    key, rk = jax.random.split(key)
    obs, env_st = env.reset(rk, env_params)
    carry = zero_carry()
    prev_action = zero_prev_action()
    checked = jnp.zeros((K,), dtype=jnp.float32)

    def step_fn(scan_carry, _):
        obs, env_st, carry, prev_action, checked, key = scan_carry
        goodness = env_st.rock_morality.astype(jnp.float32)   # the hidden state
        sampled = env_st.sampled_rocks.astype(jnp.float32)

        key, sk, ek, eps_key = jax.random.split(key, 4)
        raw, new_carry = fns.predict(agent_state, obs, carry, sk,
                                     deterministic=False, prev_action=prev_action)
        policy_action = jnp.round(raw).astype(jnp.int32)
        random_action = jax.random.randint(eps_key, policy_action.shape, 0, NUM_ACTIONS)
        use_random = jax.random.uniform(eps_key) < args.collect_epsilon
        action = jnp.where(use_random, random_action, policy_action)
        new_prev_action = (jax.nn.one_hot(action, NUM_ACTIONS)
                           if args.use_prev_action else prev_action)

        # a check of rock i is action 5 + i; remember which rocks were ever read
        is_check = action > 4
        checked_now = jnp.where(
            is_check, jnp.maximum(checked, jax.nn.one_hot(action - 5, K)), checked)

        next_obs, next_st, _, done, _ = env.step(ek, env_st, action, env_params)
        carry_out = jnp.where(done, zero_carry(), new_carry)
        prev_out = jnp.where(done, jnp.zeros_like(new_prev_action), new_prev_action)
        checked_out = jnp.where(done, jnp.zeros_like(checked_now), checked_now)
        return (next_obs, next_st, carry_out, prev_out, checked_out, key), {
            "carries": new_carry[:CARRY_DIM],
            "goodness": goodness,
            "sampled": sampled,
            "checked": checked,           # BEFORE this step's check
            "dones": done.astype(jnp.float32),
        }

    _, data = jax.lax.scan(
        step_fn, (obs, env_st, carry, prev_action, checked, key), length=n_steps)
    return data


# ── probe ────────────────────────────────────────────────────────────────────
#
# Two heads over the K rocks, the same shape Battleship uses: the hidden state
# (goodness) and what the memory should know about its own history (sampled).
# `checked` splits the metrics into retention (the sensor read this rock at some
# point) and inference (it never did).

_train_probe, _train_probe_v, _probe_chunk = common.make_probe_trainer(
    optax.adam(args.probe_lr), carry_dim=CARRY_DIM, n_out=2 * K,
    hidden=args.probe_hidden_dim, batch_size=args.probe_batch_size,
    tqdm=tqdm, wandb=_wandb)
_probe_forward_v = jax.vmap(probe_forward)
_probe_metrics_from_probs = partial(common.probe_metrics_from_probs, n_cells=K)


def _targets(goodness, sampled):
    return jnp.concatenate([jnp.asarray(goodness), jnp.asarray(sampled)], axis=-1)


def _collect_v(states_b, keys, n_steps):
    return jax.vmap(lambda s, k: collect_rollout(s, k, n_steps))(states_b, keys)


def _probe_eval_compute(states_b, gidxs, rnd):
    if args.save_fe_every_eval:
        for j, gi in enumerate(gidxs):
            _save_fe(common.unstack_state(states_b, j), gi, rnd * args.train_steps)
    ne = args.probe_eval_collect_steps
    ck_tr = jnp.stack([jax.random.key(args.seed + 500000 + rnd + gi) for gi in gidxs])
    ck_te = jnp.stack([jax.random.key(args.seed + 600000 + rnd + gi) for gi in gidxs])
    tr, te = _collect_v(states_b, ck_tr, ne), _collect_v(states_b, ck_te, ne)
    ik = jnp.stack([jax.random.key(args.seed + 40000 + rnd + gi) for gi in gidxs])
    tk = jnp.stack([jax.random.key(args.seed + 50000 + rnd + gi) for gi in gidxs])
    params_b = _train_probe_v(jnp.asarray(tr["carries"]),
                              _targets(tr["goodness"], tr["sampled"]),
                              ik, tk, args.probe_eval_steps)
    te_c = jnp.asarray(te["carries"])
    preds = np.array(jax.nn.sigmoid(_probe_forward_v(params_b, te_c)))
    te_g = np.array(te["goodness"]); te_ch = np.array(te["checked"])
    te_d = np.array(te["dones"]); te_c = np.array(te_c)
    per_seed = [
        _probe_metrics_from_probs(preds[j], te_g[j], te_ch[j],
                                  common.episode_bounds(te_d[j], len(te_c[j])))
        for j in range(len(gidxs))]
    return {"gidxs": list(gidxs), "rnd": rnd, "per_seed": per_seed}


def _probe_eval_render(host):
    gidxs, rnd, per_seed = host["gidxs"], host["rnd"], host["per_seed"]
    step = rnd * args.train_steps
    payload = {"env_interactions": step}
    for k in per_seed[0]:
        payload.update(common.agg([m[k] for m in per_seed], f"probe_eval/agg/{k}"))
        for j, gi in enumerate(gidxs):
            payload[f"seed_{gi}/probe_eval/{k}"] = per_seed[j][k]

    def _p(k, fmt=".3f"):
        return f"{payload[f'probe_eval/agg/{k}/mean']:{fmt}}"

    print(f"  [probe-eval] checked AUROC={_p('fired_auroc')}  "
          f"unchecked AUROC={_p('unfired_auroc')}  "
          f"errors/state={_p('errors_per_state')}  "
          f"horizon={_p('horizon_steps', '.0f')}  "
          f"sampled-head AUROC={_p('fired_pred_auroc')}")
    if _wandb is not None:
        _wandb.log(payload)


# ── multi-seed training ──────────────────────────────────────────────────────

CONCURRENT = args.concurrent_seeds or args.num_seeds
if args.num_seeds % CONCURRENT != 0:
    sys.exit(f"--num-seeds ({args.num_seeds}) must be divisible by "
             f"--concurrent-seeds ({CONCURRENT}).")
seeds = [args.seed + i for i in range(args.num_seeds)]
n_groups = args.num_seeds // CONCURRENT
print(f"{args.num_seeds} seed(s) in {n_groups} group(s) of {CONCURRENT}; "
      f"{args.rounds} rounds × {args.train_steps} steps.")

PREFILL_STEPS = hp.batch_size * (
    hp.lambda_truncation + hp.sequence_length + hp.burn_in_length)
_reset_v = jax.jit(jax.vmap(lambda k: env.reset(k, env_params)))
_prefill_v = jax.jit(
    jax.vmap(lambda s, es, k: fns.prefill_buffer(s, env, env_params, es,
                                                 PREFILL_STEPS, k),
             in_axes=(0, 0, 0)), donate_argnums=(0, 1))
if args.offline:
    def _offline_round(s, es, ec, k):
        s, m = fns.update_only(s, args.train_steps, k)
        return s, es, ec, m
    _train_v = jax.jit(jax.vmap(_offline_round, in_axes=(0, 0, 0, 0)),
                       donate_argnums=(0,))
else:
    _train_v = jax.jit(
        jax.vmap(lambda s, es, ec, k: fns.train_unrolled(s, env, env_params, es, ec, k),
                 in_axes=(0, 0, 0, 0)), donate_argnums=(0, 1))


def _evaluate_v(states_b, keys, n):
    return jax.vmap(lambda s, k: evaluate(s, k, n_episodes=n))(states_b, keys)


return_hist = {gi: [] for gi in range(args.num_seeds)}


def run_group(group_idx, gidxs):
    svals = [seeds[gi] for gi in gidxs]
    print(f"\n{'=' * 60}\nGroup {group_idx + 1}/{n_groups}  seeds={svals}\n{'=' * 60}")
    states = [_transplant_fe(state if gi == 0 else _build_agent(seeds[gi])[0])
              for gi in gidxs]
    batched = common.stack_states(states)
    keys = jnp.stack([jax.random.key(sv) for sv in svals])
    keys, reset_keys = common.split_each(keys)
    _obs, env_state = _reset_v(reset_keys)
    keys, prefill_keys = common.split_each(keys)
    print(f"  prefilling {PREFILL_STEPS} steps/seed…")
    batched, env_state = _prefill_v(batched, env_state, prefill_keys)
    zero_carry_b = jnp.zeros((len(gidxs), CARRY_DIM), dtype=jnp.float32)

    def round_eval(keys, batched):
        keys, eval_keys = common.split_each(keys)
        returns, steps, done = _evaluate_v(batched, eval_keys, 20)
        return keys, {"return": np.array(returns), "steps": np.array(steps),
                      "done_frac": np.array(done)}

    def on_round(rnd, step, evals, metrics):
        returns, steps = evals["return"], evals["steps"]
        for j, gi in enumerate(gidxs):
            return_hist[gi].append(float(returns[j]))
        print(f"Round {rnd:4d}/{args.rounds}  "
              f"return={float(returns.mean()):7.2f}±{float(returns.std()):.2f}  "
              f"steps={float(steps.mean()):6.1f}")
        if _wandb is None:
            return
        payload = {"round": rnd, "env_interactions": step}
        payload.update(common.agg(returns, "agg/return"))
        payload.update(common.agg(steps, "agg/steps"))
        for mk, mv in metrics.items():
            payload.update(common.agg(np.array(mv), f"agg/{mk}"))
        for j, gi in enumerate(gidxs):
            payload[f"seed_{gi}/agent/mean_return"] = float(returns[j])
        _wandb.log(payload)

    batched, keys = common.run_seed_group(
        rounds=args.rounds, train_steps=args.train_steps, keys=keys,
        batched=batched, env_state=env_state, zero_carry_b=zero_carry_b,
        train_v=_train_v, round_eval=round_eval, on_round=on_round,
        probe=common.Hooks(lambda b, rnd: _probe_eval_compute(b, gidxs, rnd),
                           _probe_eval_render),
        probe_interval=args.probe_eval_interval,
        tqdm=tqdm, desc=f"Group {group_idx + 1}")
    return batched


indexed = list(range(args.num_seeds))
groups = [indexed[g * CONCURRENT:(g + 1) * CONCURRENT] for g in range(n_groups)]
seed_states = [None] * args.num_seeds
if not args.skip_train:
    for g, gidxs in enumerate(groups):
        batched = run_group(g, gidxs)
        for j, gi in enumerate(gidxs):
            seed_states[gi] = common.unstack_state(batched, j)
            leaves, treedef = jax.tree.flatten(seed_states[gi])
            with open(os.path.join(args.output_dir, f"agent_seed{gi}.pkl"), "wb") as f:
                pickle.dump({"leaves": [np.array(l) for l in leaves],
                             "treedef": treedef}, f)
    W = max(1, args.final_return_window)
    fin = np.array([np.mean(return_hist[gi][-W:]) for gi in range(args.num_seeds)])
    agg_final = common.agg(fin, "final/return_smoothed")
    print(f"\n{'=' * 60}\nAggregated over {args.num_seeds} seed(s):\n"
          f"  final/return_smoothed = {agg_final['final/return_smoothed/mean']:.2f} "
          f"± {agg_final['final/return_smoothed/sterr']:.2f} (sterr)\n"
          f"  (exit-only floor is {ENV_CONFIG['exit_reward']:.0f})\n{'=' * 60}")
    if _wandb is not None:
        _wandb.log(agg_final)
        _wandb.summary.update(agg_final)
else:
    for gi in range(args.num_seeds):
        with open(os.path.join(args.output_dir, f"agent_seed{gi}.pkl"), "rb") as f:
            saved = pickle.load(f)
        _, treedef = jax.tree.flatten(state)
        seed_states[gi] = treedef.unflatten([jnp.array(l) for l in saved["leaves"]])

if args.train_only:
    print("--train-only: skipping the final probe.")
    if _wandb is not None:
        _wandb.finish()
    sys.exit(0)

# ── final probe on held-out rollouts ─────────────────────────────────────────

all_states = common.stack_states(seed_states)
print(f"Collecting {args.collect_steps} train+test steps per seed…")
ck_tr = jnp.stack([jax.random.key(seeds[gi] + 1000) for gi in range(args.num_seeds)])
ck_te = jnp.stack([jax.random.key(seeds[gi] + 2000) for gi in range(args.num_seeds)])
tr, te = _collect_v(all_states, ck_tr, args.collect_steps), \
    _collect_v(all_states, ck_te, args.collect_steps)

print(f"Training probe per seed ({args.probe_steps} steps)…")
ik = jnp.stack([jax.random.key(seeds[gi] + 20000) for gi in range(args.num_seeds)])
tk = jnp.stack([jax.random.key(seeds[gi] + 30000) for gi in range(args.num_seeds)])
probe_b = _train_probe_v(jnp.asarray(tr["carries"]),
                         _targets(tr["goodness"], tr["sampled"]),
                         ik, tk, args.probe_steps)
preds = np.array(jax.nn.sigmoid(_probe_forward_v(probe_b, jnp.asarray(te["carries"]))))
te_g, te_ch = np.array(te["goodness"]), np.array(te["checked"])
te_d, te_c = np.array(te["dones"]), np.array(te["carries"])
per_seed = [_probe_metrics_from_probs(preds[j], te_g[j], te_ch[j],
                                      common.episode_bounds(te_d[j], len(te_c[j])))
            for j in range(args.num_seeds)]
final_payload = {}
for k in per_seed[0]:
    final_payload.update(common.agg([m[k] for m in per_seed], f"eval/agg/{k}"))
    for gi in range(args.num_seeds):
        final_payload[f"eval/seed_{gi}/{k}"] = per_seed[gi][k]
print(f"  checked (retention)  AUROC = {final_payload['eval/agg/fired_auroc/mean']:.3f} "
      f"± {final_payload['eval/agg/fired_auroc/sterr']:.3f}")
print(f"  unchecked (inference) AUROC = {final_payload['eval/agg/unfired_auroc/mean']:.3f}")
print(f"  sampled head          AUROC = {final_payload['eval/agg/fired_pred_auroc/mean']:.3f}")
if _wandb is not None:
    _wandb.log(final_payload)
    _wandb.summary.update(final_payload)
    _wandb.finish()
print("Done.")
