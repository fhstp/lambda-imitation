"""Offline experiment: train the Battleship agent on a Bayes player's data only.

The online runs face a chicken-and-egg problem — the memory only pays off once
the policy hunts, and the policy only hunts once the memory works — so the
board never gets encoded (see ``ablations.md``).  This script removes the
policy side of that loop entirely:

  1. a scripted **Bayes-density player** fills the replay buffer (no learning),
  2. the agent runs ``--updates`` gradient steps on that buffer with **zero
     environment interaction**,
  3. a probe then asks whether the recurrent memory encodes the hidden board.

If λ-discrepancy (or GVD) can form board memory *at all*, it should form it
here: the data is what a good player generates, so hit locations are
value-relevant from the first update.  A flat probe here is evidence the
memory-formation gap is in the objective, not in exploration.

The Bayes player is the baseline from the return-ceiling calculation: count
every ship placement consistent with the observed misses, weight it by
``--hit-weight`` per observed hit it covers, fire at the highest-density
unfired cell.  On 5x5 with ships (3,2) it clears in ~13.3 shots (return ~12.7)
against ~21.7 for uniform random.

Everything except the offline loop and the Bayes policy is imported from
``battleship_board_probe.py``: the env, the agent, the probe, the metrics and
the figures all come from there (it is loaded with ``--setup-only``, which
builds its globals and exits before its own Phase 1).  Every flag of that
script therefore works here too.

Usage:
    python battleship_offline_bayes.py                       # 5x5, ships 3,2
    python battleship_offline_bayes.py --offline-epsilon 0    # pure greedy data
    python battleship_offline_bayes.py --no-approximate-lambda  # SAC baseline
    python battleship_offline_bayes.py --updates 100000 --wandb
"""

import argparse
import importlib.util
import os
import pickle
import sys

import numpy as np

# ── CLI (offline-specific; every other flag is parsed by the probe module) ───

parser = argparse.ArgumentParser(
    description="Offline Bayes-data training + memory probe for Battleship.",
    epilog="All battleship_board_probe.py flags (--rows, --memory-type, "
           "--gvd, --wandb, …) are accepted and forwarded.",
)
parser.add_argument("--updates", type=int, default=50_000,
                    help="total gradient updates on the offline buffer (default 50 000)")
parser.add_argument("--update-chunk", type=int, default=1_000,
                    help="updates per logged round (default 1 000)")
parser.add_argument("--offline-fill", type=int, default=100_000,
                    help="transitions of Bayes-player data to fill the buffer "
                         "with (default 100 000; capped at --online-buffer-size)")
parser.add_argument("--offline-epsilon", type=float, default=0.1,
                    help="epsilon-greedy rate of the Bayes behaviour policy "
                         "(default 0.1; 0 = pure greedy)")
parser.add_argument("--hit-weight", type=float, default=12.0,
                    help="posterior weight per covered hit in the density "
                         "player (default 12; results are flat over 2…1000)")
parser.add_argument("--probe-policies", default="actor,bayes",
                    help="which rollout policies to probe, comma-separated "
                         "(default 'actor,bayes' — the learned actor and the "
                         "Bayes player whose data it trained on)")
parser.add_argument("--policy-check-episodes", type=int, default=200,
                    help="episodes used for the startup sanity check of the "
                         "Bayes policy (0 disables; default 200)")
args, _ = parser.parse_known_args()

# ── load battleship_board_probe.py for its env / agent / probe machinery ─────
#
# That script runs at import (argparse at module level, phases after it), so it
# is loaded with --setup-only appended: it builds env, hp, the agent and every
# collect/probe/figure helper, then raises SystemExit before its Phase 1.  We
# keep the half-executed module object — everything defined up to that exit is
# on it.  Its parser uses parse_known_args, so our own flags pass through
# harmlessly and its flags are parsed from the same argv.

_PROBE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "battleship_board_probe.py")
sys.argv = [*sys.argv, "--setup-only"]
_spec = importlib.util.spec_from_file_location("battleship_board_probe", _PROBE_PATH)
probe = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(probe)
except SystemExit:
    pass   # expected: --setup-only hands control back here
if not hasattr(probe, "fns"):
    sys.exit("battleship_board_probe.py exited before building the agent — "
             "check that --setup-only is still wired before its Phase 1.")

import jax                      # noqa: E402  (after probe sets up JAX config)
import jax.numpy as jnp         # noqa: E402
from tqdm import tqdm           # noqa: E402

env, env_params = probe.env, probe.env_params
fns, state = probe.fns, probe.state
N, ROWS, COLS = probe.N, probe.args.rows, probe.args.cols
SHIP_LENGTHS = tuple(int(x) for x in probe.args.ship_lengths.split(",") if x.strip())
_MAX_STEPS = probe._MAX_STEPS

if "--output-dir" not in sys.argv:
    probe.args.output_dir = "./battleship_offline_bayes_output"
OUT = probe.args.output_dir
os.makedirs(OUT, exist_ok=True)


# ── the Bayes-density player ─────────────────────────────────────────────────


def make_bayes_policy(rows, cols, ship_lengths, hit_weight=12.0, epsilon=0.1):
    """Posterior-density greedy Battleship player, jittable.

    Returns ``policy(obs, env_state, key) -> (action_index, b(a|s))``, the
    contract taken by ``fns.prefill_buffer(behaviour_fn=…)`` and by the probe
    script's ``collect_rollout(policy_fn=…)``.

    The density counts, for every ship length, all placements that do not touch
    an observed miss, weighting each by ``hit_weight ** (#observed hits it
    covers)`` — the standard stand-in for the joint posterior when the env
    gives no sink feedback — and scatters that weight onto its cells.  The
    player then fires at the highest-density unfired cell, or with probability
    ``epsilon`` at a uniformly random legal one.

    It reads ``env_state.hits_misses``, which is the player's *own* shot record
    (0 unfired / 1 miss / 2 hit) — the same information the observation stream
    carries — never ``env_state.board``.
    """
    n = rows * cols
    grid = np.arange(n).reshape(rows, cols)
    placements = []
    for length in sorted(set(ship_lengths)):
        horiz = [grid[r, c:c + length]
                 for r in range(rows) for c in range(cols - length + 1)]
        vert = [grid[r:r + length, c]
                for c in range(cols) for r in range(rows - length + 1)]
        placements.append(jnp.asarray(np.stack(horiz + vert)))   # (n_place, L)

    def policy(obs, env_state, key):
        hm = env_state.hits_misses.reshape(-1)          # 0 unfired, 1 miss, 2 hit
        density = jnp.zeros(n, dtype=jnp.float32)
        for place in placements:                        # static: one per length
            cells = hm[place]                           # (n_place, L)
            alive = ~jnp.any(cells == 1, axis=-1)       # no miss under the ship
            weight = hit_weight ** jnp.sum(cells == 2, axis=-1) * alive
            density = density.at[place.reshape(-1)].add(
                jnp.repeat(weight, place.shape[1]))

        legal = hm == 0
        greedy = jnp.argmax(jnp.where(legal, density, -1.0))
        pick_key, coin_key = jax.random.split(key)
        random_legal = jax.random.categorical(
            pick_key, jnp.where(legal, 0.0, -1e9))
        explore = jax.random.uniform(coin_key) < epsilon
        action = jnp.where(explore, random_legal, greedy)

        n_legal = jnp.maximum(jnp.sum(legal), 1).astype(jnp.float32)
        prob = jnp.where(action == greedy,
                         (1.0 - epsilon) + epsilon / n_legal,
                         epsilon / n_legal)
        return action.astype(jnp.int32), prob.astype(jnp.float32)

    return policy


def evaluate_policy(policy, key, n_episodes):
    """Mean shots-to-clear and mean return of a scripted policy (no agent)."""

    def run_episode(k):
        k, reset_key = jax.random.split(k)
        _obs, env_st = env.reset(reset_key, env_params)

        def step_fn(carry, _):
            env_st, k, ret, done, steps = carry
            k, ak, ek = jax.random.split(k, 3)
            obs = env.get_obs(env_st, env_params)
            action, _prob = policy(obs, env_st, ak)
            _nobs, env_st, reward, d, _ = env.step(ek, env_st, action, env_params)
            ret = ret + reward * (1.0 - done)
            steps = steps + (1.0 - done)
            done = jnp.maximum(done, d.astype(jnp.float32))
            return (env_st, k, ret, done, steps), None

        init = (env_st, k, jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0))
        (_, _, ret, _, steps), _ = jax.lax.scan(step_fn, init, length=_MAX_STEPS)
        return ret, steps

    rets, steps = jax.vmap(run_episode)(jax.random.split(key, n_episodes))
    return float(jnp.mean(steps)), float(jnp.mean(rets))


bayes_policy = make_bayes_policy(
    ROWS, COLS, SHIP_LENGTHS, hit_weight=args.hit_weight,
    epsilon=args.offline_epsilon,
)
greedy_policy = make_bayes_policy(
    ROWS, COLS, SHIP_LENGTHS, hit_weight=args.hit_weight, epsilon=0.0,
)


def _random_policy(obs, env_state, key):
    legal = env_state.hits_misses.reshape(-1) == 0
    action = jax.random.categorical(key, jnp.where(legal, 0.0, -1e9))
    n_legal = jnp.maximum(jnp.sum(legal), 1).astype(jnp.float32)
    return action.astype(jnp.int32), (1.0 / n_legal).astype(jnp.float32)


# ── sanity check: the scripted player must actually play well ────────────────

if args.policy_check_episodes > 0:
    ck = jax.random.key(probe.args.seed + 90_000)
    g_shots, g_ret = evaluate_policy(greedy_policy, ck, args.policy_check_episodes)
    b_shots, b_ret = evaluate_policy(bayes_policy, ck, args.policy_check_episodes)
    r_shots, r_ret = evaluate_policy(_random_policy, ck, args.policy_check_episodes)
    print(f"Bayes player on {ROWS}x{COLS}, ships {SHIP_LENGTHS} "
          f"({args.policy_check_episodes} episodes):")
    print(f"  greedy (eps=0)      : {g_shots:5.2f} shots  return {g_ret:6.2f}")
    print(f"  behaviour (eps={args.offline_epsilon:g}) : "
          f"{b_shots:5.2f} shots  return {b_ret:6.2f}")
    print(f"  uniform random      : {r_shots:5.2f} shots  return {r_ret:6.2f}")
    assert g_shots < 0.8 * r_shots, (
        f"Bayes policy is not playing well: {g_shots:.2f} shots vs "
        f"{r_shots:.2f} random — the density computation is wrong."
    )


# ── Phase 1 — fill the buffer with Bayes data, then update offline ───────────

FILL = min(args.offline_fill, probe.args.online_buffer_size)
rounds = max(1, args.updates // args.update_chunk)
tag = f"{probe.tag} offline-bayes(eps={args.offline_epsilon:g})"

key = jax.random.key(probe.args.seed)
key, reset_key, fill_key = jax.random.split(key, 3)
_obs, env_state = env.reset(reset_key, env_params)

print(f"Filling buffer with {FILL} Bayes-player transitions "
      f"(eps={args.offline_epsilon:g})…")
state, env_state = fns.prefill_buffer(
    state, env, env_params, env_state, FILL, fill_key, behaviour_fn=bayes_policy
)
n_sampleable = int(state.online_buffer.sampling_ok.sum())
print(f"  {n_sampleable} sampleable transitions in the buffer.")

def run_probe_eval(agent_state, rnd):
    """Lightweight probe on the current memory, along both rollout policies.

    ``rnd == 0`` is the anchor: an untrained FE, whose decodability is the
    floor every later point is read against.
    """
    step = rnd * args.update_chunk
    for name, pol in (("actor", None), ("bayes", bayes_policy)):
        c, b, hm, _eb = probe._collect_and_parse(
            agent_state, 5000 + rnd, probe.args.probe_eval_collect_steps, policy_fn=pol)
        tc, tb, thm, _teb = probe._collect_and_parse(
            agent_state, 6000 + rnd, probe.args.probe_eval_collect_steps, policy_fn=pol)
        params = probe._train_probe(
            jnp.array(c), probe._targets_from(b, hm),
            jax.random.key(probe.args.seed + 40_000 + rnd),
            jax.random.key(probe.args.seed + 50_000 + rnd),
            probe.args.probe_eval_steps,
        )
        m = probe._probe_metrics(params, tc, tb, thm)
        eval_history.append({"updates": step, "rollouts": name, **m})
        tqdm.write(f"  [probe-eval/{name} @ {step}] fired AUROC={m['fired_auroc']:.3f}  "
                   f"unfired AUROC={m['unfired_auroc']:.3f}  "
                   f"fired bal={m['fired_balanced']:.1%}  "
                   f"unfired bal={m['unfired_balanced']:.1%}  "
                   f"overall={m['overall_acc']:.1%}")
        if probe._wandb is not None:
            probe._wandb.log({"offline_updates": step,
                              **{f"probe_eval_{name}/{k}": v for k, v in m.items()}})


history, eval_history = [], []
if probe.args.probe_eval_interval:
    print("Probing the untrained memory (update 0 anchor)…")
    run_probe_eval(state, 0)

print(f"Training {rounds} × {args.update_chunk} = {rounds * args.update_chunk} "
      f"offline updates (no env interaction)…")
for rnd in tqdm(range(1, rounds + 1), desc="Offline updates"):
    key, update_key, eval_key = jax.random.split(key, 3)
    state, metrics = fns.update_only(state, args.update_chunk, update_key)
    mean_ret, steps_to_clear, cleared = probe.evaluate(state, eval_key, n_episodes=10)
    row = {
        "updates": rnd * args.update_chunk,
        "return": float(mean_ret),
        "steps_to_clear": float(steps_to_clear),
        "cleared": float(cleared),
        **{k: float(v) for k, v in metrics.items()},
    }
    history.append(row)
    tqdm.write(
        f"  [{row['updates']:>7d}] return={row['return']:7.2f}  "
        f"steps={row['steps_to_clear']:5.1f}  cleared={row['cleared']:.2f}  "
        f"critic_loss={row.get('critic_loss', float('nan')):.4f}"
        # ld_mean = constant offset between the λ-critics, ld_std = the
        # state-dependent part; the offset was the whole term before.
        + (f"  ld={row['ld_loss']:.3g} (mean={row['ld_mean']:.3g} "
           f"std={row['ld_std']:.3g})" if "ld_std" in row else "")
        + (f"  Eq1={row['lambda0.05_critic:']:.1f} Eq2={row['lambda0.85_critic:']:.1f}"
           if "lambda0.05_critic:" in row else "")
        # prioritised replay: mean window trace mass, and the effective sample
        # size of the batch (1.0 = uniform, →0 = the batch collapsed onto a few
        # windows).
        + (f"  per(pri={row['per_priority']:.3g}±{row['per_priority_std']:.3g} "
           f"ess={row['per_ess']:.2f})" if "per_ess" in row else "")
    )
    if probe._wandb is not None:
        probe._wandb.log({"offline_updates": row["updates"],
                          **{f"offline/{k}": v for k, v in row.items()}})

    if probe.args.probe_eval_interval and rnd % probe.args.probe_eval_interval == 0:
        run_probe_eval(state, rnd)

agent_path = os.path.join(OUT, "agent.pkl")
print(f"Saving agent → {agent_path}")
_leaves, _treedef = jax.tree.flatten(state)
with open(agent_path, "wb") as f:
    pickle.dump({"leaves": [np.array(l) for l in _leaves], "treedef": _treedef}, f)
with open(os.path.join(OUT, "offline_history.pkl"), "wb") as f:
    pickle.dump({"history": history, "eval_history": eval_history, "tag": tag,
                 "fill": FILL, "epsilon": args.offline_epsilon}, f)


# ── Phase 2/3 — probe the memory, once per rollout policy ────────────────────

POLICIES = {"actor": None, "bayes": bayes_policy}
wanted = [p.strip() for p in args.probe_policies.split(",") if p.strip()]
unknown = [p for p in wanted if p not in POLICIES]
if unknown:
    sys.exit(f"unknown --probe-policies entries {unknown}; pick from {list(POLICIES)}")

results = {}
for name in wanted:
    pol = POLICIES[name]
    vdir = os.path.join(OUT, name)
    os.makedirs(vdir, exist_ok=True)
    print(f"\n── probing along {name} rollouts "
          f"({probe.args.collect_steps} steps train + test) ──")
    c, b, hm, eb = probe._collect_and_parse(
        state, 1000, probe.args.collect_steps, policy_fn=pol)
    tc, tb, thm, teb = probe._collect_and_parse(
        state, 2000, probe.args.collect_steps, policy_fn=pol)
    print(f"  train {len(c)} steps / {len(eb) - 1} eps, "
          f"test {len(tc)} steps / {len(teb) - 1} eps")
    with open(os.path.join(vdir, "dataset.pkl"), "wb") as f:
        pickle.dump({"carries": c, "board_masks": b, "hits_misses": hm,
                     "ep_bounds": eb, "rows": ROWS, "cols": COLS, "tag": tag}, f)
    with open(os.path.join(vdir, "test_dataset.pkl"), "wb") as f:
        pickle.dump({"carries": tc, "board_masks": tb, "hits_misses": thm,
                     "ep_bounds": teb, "rows": ROWS, "cols": COLS, "tag": tag}, f)

    params = probe._train_probe(
        jnp.array(c), probe._targets_from(b, hm),
        jax.random.key(probe.args.seed + 70_000),
        jax.random.key(probe.args.seed + 80_000),
        probe.args.probe_steps, verbose=True,
    )
    with open(os.path.join(vdir, "probe.pkl"), "wb") as f:
        pickle.dump(jax.tree.map(np.asarray, params), f)

    m = probe._probe_metrics(params, tc, tb, thm)
    results[name] = m
    vtag = f"{tag} [{name} rollouts]"
    n_vis = min(probe.args.vis_episodes, len(teb) - 1)
    for vi in range(n_vis):
        probe._fig_episode(params, tc, tb, thm, teb, vi,
                           os.path.join(vdir, f"board_probe_ep{vi + 1}.png"),
                           vtag, probe.args.vis_frames)
    probe._fig_accuracy(params, tc, tb, thm,
                        os.path.join(vdir, "board_probe_accuracy.png"), vtag)
    probe._fig_retention(params, tc, tb, thm, teb,
                         os.path.join(vdir, "board_probe_retention.png"), vtag)
    if probe._wandb is not None:
        probe._wandb.log({f"probe_{name}/{k}": v for k, v in m.items()})


# ── summary ──────────────────────────────────────────────────────────────────

print(f"\n{tag}: {rounds * args.update_chunk} offline updates on {FILL} "
      f"Bayes transitions, 0 env steps during training.")
print(f"{'rollouts':<10}{'fired AUROC':>13}{'unfired AUROC':>15}"
      f"{'fired bal':>11}{'unfired bal':>13}{'overall':>9}")
for name, m in results.items():
    print(f"{name:<10}{m['fired_auroc']:>13.3f}{m['unfired_auroc']:>15.3f}"
          f"{m['fired_balanced']:>11.1%}{m['unfired_balanced']:>13.1%}"
          f"{m['overall_acc']:>9.1%}")
print("(AUROC 0.5 = nothing decodable; 'fired' = retention of what was "
      "observed, 'unfired' = inference of the hidden board.)")
print(f"Artefacts → {OUT}/<rollout-policy>/")
if probe._wandb is not None:
    probe._wandb.finish()
