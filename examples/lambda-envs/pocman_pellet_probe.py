"""Pellet probe for PocMan: train a probe (memory → remaining pellets) and visualise.

PocMan's observation is 11 bits — four wall sensors, a food smell, a ghost
hearing bit, four line-of-sight bits and the power-pill flag.  It never carries
the agent's position, so to know which parts of the maze it has already cleared
the recurrent memory must integrate the whole action–observation history.  The
probe asks exactly that: decode the remaining-pellet map from the carry.

This is the PocMan counterpart of ``battleship_board_probe.py`` and shares its
interface: both scripts import ``_probe_common`` for the CLI, the probe MLP, the
metrics, the multi-seed loop and the W&B wiring, so every flag means the same
thing in both.  What differs is the env, what the probe decodes, and the
drawing.

Pipeline (multi-seed throughout — ``--num-seeds 1`` is just a group of one):
  1. Train a PocMan SAC(+LD) agent (recurrent memory, prev-action input)
  2. Collect rollouts saving hidden states + the true pellet map
  3. Train an MLP probe: hidden state → per-pellet occupancy (BCE)
  4. Visualise probe predictions against the true maze

Unlike Battleship, a pellet is gone *because* the agent ate it, so "what is
hidden" and "what has been observed" are complements: there is no inference
half.  The question is retention — how long the memory keeps a cleared cell —
so the metrics break down by RECENCY (see ``probe_metrics_visitation``).

Usage:
    python pocman_pellet_probe.py                               # full pipeline
    python pocman_pellet_probe.py --num-seeds 3 --wandb         # 3 seeds, logged
    python pocman_pellet_probe.py --expert-prefill-steps 20000  # expert warm start
    python pocman_pellet_probe.py --offline --expert-prefill-steps 100000
    python pocman_pellet_probe.py --vis-only --mp4              # render only
"""

import argparse
import os
import pickle
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_common as common

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="PocMan pellet probe visualisation.")

g = parser.add_argument_group("environment")
g.add_argument("--eval-max-steps", type=int, default=1000,
               help="episode cap for the evaluation / collection scans (default "
                    "1000 = the env's own time limit).  Lower it to trade eval "
                    "fidelity for speed; the scripted expert needs ~460 steps.")

g = parser.add_argument_group("network architecture")
g.add_argument("--paper-arch", dest="paper_arch", action="store_true",
               help="use the original lambda-discrepancy PocMan network "
                    "(DiscreteActorCriticRNN in lamb/models.py): a "
                    "Dense(H)->ReLU embedding over [obs | prev-action], a "
                    "GRU(H) memory, and single-hidden-layer actor/critic heads "
                    "of width H, with no critic LayerNorm.  H is "
                    "--memory-hidden-dim (the paper uses 512).  Overrides "
                    "--projection-dim, the head widths and --critic-layer-norm. "
                    "Default on; --no-paper-arch restores the previous "
                    "(256, 256) heads with LayerNorm.")
g.add_argument("--no-paper-arch", dest="paper_arch", action="store_false")
parser.set_defaults(paper_arch=True)
g.add_argument("--lambda1", type=float, default=0.5,
               help="lambda of the first lambda-critic (default 0.5)")
g.add_argument("--lambda2", type=float, default=0.95,
               help="lambda of the second lambda-critic (default 0.95).  The "
                    "pair (0.5, 0.95) is what the reference implementation "
                    "selected for PocMan (pocman_LD_ppo_best.py).")

g = parser.add_argument_group("scripted expert (pocman_expert.py)")
g.add_argument("--expert-safety-margin", type=int, default=5,
               help="the expert never steps within this many cells of a live ghost")
g.add_argument("--expert-rollout-depth", type=int, default=28,
               help="rollout-policy-improvement depth of the expert (0 = greedy "
                    "table policy; 28 is the measured plateau)")
g.add_argument("--expert-no-chase", action="store_true",
               help="expert ignores edible ghosts while a power pill is active")

common.add_common_args(
    parser, output_dir_default="./pocman_probe_output",
    wandb_project_default="offline-lambda-pocman-results")

# PocMan-tuned defaults (the values the earlier PocMan runs used); every one of
# these is still a flag, this only changes what you get without passing it.
parser.set_defaults(
    rounds=8, gamma=0.95, tau=0.006,
    fe_lr=7e-5, actor_lr=6e-5, critic_lr=2e-4,
    memory_hidden_dim=512, batch_size=512,   # 512 = the paper's hidden_size
    sequence_length=20, burn_in_length=32, online_buffer_size=200_000,
)

args = common.parse_args(parser, extra_flags_env="POCMAN_PROBE_EXTRA_FLAGS")
_SWEEP_RUN = common.apply_sweep_config(parser, args)

# ── always-needed imports ────────────────────────────────────────────────────

import jax
import jax.numpy as jnp
import numpy as np

# ── wandb (optional) ─────────────────────────────────────────────────────────

_wandb = common.init_wandb(args, {
    "env": "PocMan",
    "experiment": "pellet_probe",
    "algo": ("SAC+LD" if args.approximate_lambda else "SAC") + ("+GVD" if args.gvd else ""),
    "eval_max_steps": args.eval_max_steps,
    "expert_safety_margin": args.expert_safety_margin,
    "expert_rollout_depth": args.expert_rollout_depth,
    "paper_arch": args.paper_arch,
    "lambda1": args.lambda1,
    "lambda2": args.lambda2,
    **common.common_wandb_config(args),
}, _SWEEP_RUN)

# ── probe MLP (shared with the Battleship probe) ─────────────────────────────

init_probe_params = common.init_probe_params
probe_forward = common.probe_forward

# ── shared rendering ─────────────────────────────────────────────────────────
#
# Module level so the periodic probe-eval and the final visualisation phase draw
# identical figures.  Everything here reads the maze geometry from the dataset
# (wall grid + pellet cell coordinates), so --vis-only needs no env.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

WALL_COLOR = "#1b2430"
FLOOR_COLOR = "#0d1117"
PELLET_COLOR = "#ffd166"
PLAYER_COLOR = "#4cc9f0"
# floor → pellet for P(pellet); red → green for per-cell accuracy
CMAP_PELLET = LinearSegmentedColormap.from_list("pellet", [FLOOR_COLOR, PELLET_COLOR])
CMAP_ACC = LinearSegmentedColormap.from_list("acc", ["#c1121f", "#f6e05e", "#2d936c"])

def to_grid(values, cells, wall_grid, fill=np.nan):
    """Scatter per-pellet values onto the maze grid (walls stay ``fill``)."""
    grid = np.full(wall_grid.shape, fill, dtype=np.float32)
    grid[cells[:, 0], cells[:, 1]] = values
    return grid

def render(ax, grid, wall_grid, title="", player=None, cmap=CMAP_PELLET,
           vmin=0.0, vmax=1.0):
    """Draw one maze panel: walls, a per-cell value and (optionally) the player."""
    ax.imshow(np.where(wall_grid, 1.0, 0.0), cmap=matplotlib.colors.ListedColormap(
        [FLOOR_COLOR, WALL_COLOR]), vmin=0, vmax=1, interpolation="nearest")
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation="nearest")
    if player is not None:
        ax.plot(player[1], player[0], marker="o", markersize=4,
                color=PLAYER_COLOR, markeredgecolor="white", markeredgewidth=0.4)
    ax.set_title(title, fontsize=7, color="white")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

def _figure(nrows, ncols, size=1.6):
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(size * ncols, size * 1.15 * nrows))
    fig.patch.set_facecolor("#05070a")
    return fig, np.atleast_2d(axes)

def _save(fig, out_path, tag_str):
    fig.suptitle(tag_str, fontsize=9, color="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {out_path}")
    return out_path

def _probs_for(probe_params, carries, probs_all=None):
    """P(pellet) per step, either precomputed on host or run from params."""
    if probs_all is not None:
        return np.asarray(probs_all)
    return np.array(jax.nn.sigmoid(probe_forward(probe_params, jnp.array(carries))))

def _fig_episode(probe_params, carries, truth, players, ep_bounds, ve_idx, out_path,
                 tag_str, n_frames, cells, wall_grid, probs_all=None):
    """Ground truth vs decoded pellet map across one episode."""
    if len(ep_bounds) - 1 <= ve_idx:
        return None
    s0, s1 = int(ep_bounds[ve_idx]), int(ep_bounds[ve_idx + 1])
    if s1 - s0 < 2:
        return None
    probs = _probs_for(probe_params, carries[s0:s1], None if probs_all is None
                       else np.asarray(probs_all)[s0:s1])
    idx = np.linspace(0, s1 - s0 - 1, min(n_frames, s1 - s0)).astype(int)
    fig, axes = _figure(2, len(idx))
    for col, t in enumerate(idx):
        render(axes[0, col], to_grid(truth[s0 + t], cells, wall_grid), wall_grid,
               f"t={t}  truth", player=(players[s0 + t, 0], players[s0 + t, 1]))
        render(axes[1, col], to_grid(probs[t], cells, wall_grid), wall_grid,
               "P(pellet)", player=(players[s0 + t, 0], players[s0 + t, 1]))
    return _save(fig, out_path, f"{tag_str} — pellet memory")

def _fig_accuracy(probe_params, carries, truth, out_path, tag_str, cells, wall_grid,
                  probs_all=None):
    """Per-cell decode accuracy over the whole test set."""
    probs = _probs_for(probe_params, carries, probs_all)
    correct = ((probs > 0.5).astype(np.float32) == np.asarray(truth)).mean(axis=0)
    eaten_rate = 1.0 - np.asarray(truth).mean(axis=0)
    fig, axes = _figure(1, 2, size=2.6)
    render(axes[0, 0], to_grid(correct, cells, wall_grid), wall_grid,
           f"per-cell accuracy (mean {correct.mean():.1%})", cmap=CMAP_ACC)
    render(axes[0, 1], to_grid(eaten_rate, cells, wall_grid), wall_grid,
           "fraction of steps this cell was already eaten")
    return _save(fig, out_path, f"{tag_str} — decode accuracy")

def _fig_retention(probe_params, carries, truth, ep_bounds, out_path, tag_str,
                   probs_all=None):
    """Recall on eaten cells against how long ago they were eaten.

    This is the retention horizon: recall at age 0-2 is nearly free (the memory
    just saw it), so what distinguishes a real memory is how far right the curve
    stays high.
    """
    probs = _probs_for(probe_params, carries, probs_all)
    m = common.probe_metrics_visitation(probs, truth, ep_bounds)
    labels, values = [], []
    for lo, hi in common.AGE_BUCKETS:
        key = f"bal_age{lo}_{hi if hi < 10 ** 6 else 'plus'}"
        labels.append(f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+")
        values.append(m[key])
    if all(np.isnan(v) for v in values):
        return None
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    fig.patch.set_facecolor("#05070a"); ax.set_facecolor("#0d1117")
    ax.plot(labels, values, marker="o", color=PELLET_COLOR, label="recall (eaten cells)")
    ax.axhline(0.9, ls=":", lw=1, color="#8d99ae", label="0.9 (horizon threshold)")
    ax.axhline(m["overall_acc"], ls="--", lw=1, color=PLAYER_COLOR, label="overall accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("steps since the cell was eaten", fontsize=8)
    ax.set_ylabel("recall", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for s in ax.spines.values():
        s.set_color("#8d99ae")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.legend(fontsize=6, facecolor="#0d1117", labelcolor="white")
    return _save(fig, out_path, f"{tag_str} — retention horizon "
                                f"({m['horizon_steps']:.0f} steps)")

def _fig_movie(probe_params, carries, truth, players, out_path_base, tag_str,
               cells, wall_grid, fps=10, probs_all=None):
    """Animate one episode: truth | decoded, side by side."""
    import matplotlib.animation as animation

    probs = _probs_for(probe_params, carries, probs_all)
    fig, axes = _figure(1, 2, size=2.6)

    def draw(t):
        for ax in axes[0]:
            ax.clear()
        render(axes[0, 0], to_grid(truth[t], cells, wall_grid), wall_grid,
               f"t={t}  truth", player=(players[t, 0], players[t, 1]))
        render(axes[0, 1], to_grid(probs[t], cells, wall_grid), wall_grid,
               "P(pellet)", player=(players[t, 0], players[t, 1]))

    anim = animation.FuncAnimation(fig, draw, frames=len(carries), interval=1000 // fps)
    for ext, writer in (("mp4", "ffmpeg"), ("gif", "pillow")):
        try:
            path = f"{out_path_base}.{ext}"
            anim.save(path, writer=writer, fps=fps,
                      savefig_kwargs={"facecolor": fig.get_facecolor()})
            plt.close(fig)
            print(f"  → {path}")
            return path
        except Exception:
            continue
    plt.close(fig)
    print("  (no mp4/gif writer available)")
    return None

# ── output paths ─────────────────────────────────────────────────────────────

os.makedirs(args.output_dir, exist_ok=True)

def _seed_path(kind, gi):
    return os.path.join(args.output_dir, f"{kind}_seed{gi}.pkl")

# ════════════════════════════════════════════════════════════════════════════
#  Full pipeline (phases 1-3) — skipped entirely in --vis-only
# ════════════════════════════════════════════════════════════════════════════

if not args.vis_only:
    import optax
    from tqdm.rich import tqdm

    try:
        from lambda_envs.envs.pocman import PocMan, SMALLER_GAME_MAP
    except ImportError:
        sys.exit("lambda-envs[pocman] required.  pip install 'lambda-envs[pocman]'")

    from lambda_imitation.iqlearn import Hyperparameters
    from lambda_imitation.utils import (create_iqlearn_from_env,
                                        env_spec_from_gymnax, relu_projection)
    from pocman_expert import make_pocman_expert

    # ── env setup ────────────────────────────────────────────────────────────
    #
    # The previous action reaches the feature extractor via use_prev_action=True
    # (an explicit projection input threaded next to the carry).  Without it the
    # memory cannot localise at all: the observation is direction-symmetric, so
    # only the action stream says which way the agent went.

    class _LambdaEnvAdapter:
        """lambda-envs' PocMan.get_obs takes no params; gymnax callers pass one."""

        def __init__(self, wrapped):
            self._wrapped = wrapped

        def get_obs(self, state, params=None):
            return self._wrapped.get_obs(state)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    _raw_env = PocMan()
    env = _LambdaEnvAdapter(_raw_env)
    env_params = env.default_params
    spec = env_spec_from_gymnax(env, env_params)
    NUM_ACTIONS = 4

    if args.offline and args.expert_prefill_steps <= 0:
        sys.exit("--offline needs --expert-prefill-steps > 0: with no expert "
                 "data the buffer would hold uniform-random transitions and "
                 "there would be nothing to learn from.")
    if args.probe_rollout_policy != "actor" and args.expert_prefill_steps <= 0:
        sys.exit("--probe-rollout-policy expert needs --expert-prefill-steps > 0 "
                 "(the scripted player is only built when it is used).")

    # ── maze geometry ────────────────────────────────────────────────────────

    WALL_GRID = np.array([[c == "X" for c in row] for row in SMALLER_GAME_MAP])
    _dummy_obs, _dummy_st = jax.jit(env.reset)(jax.random.key(0), env_params)
    # pellet slots are fixed for the whole run; state arrays are (col, row)
    _pellet_locs = np.array(_dummy_st.pellet_locations)
    PELLET_CELLS = np.stack([_pellet_locs[:, 1], _pellet_locs[:, 0]], axis=-1)
    NUM_PELLETS = PELLET_CELLS.shape[0]

    expert_data = {
        "observations": jnp.zeros((1, *spec.obs_shape), dtype=jnp.float32),
        "actions": jnp.zeros((1, 1), dtype=jnp.float32),
    }

    if args.gvd:
        # φ(o_t, a_{t-1}) = the food-smell bit: a reward-free cumulant that is
        # only predictable if the memory tracks where the pellets went.
        def gvd_feature_fn(o, a_prev):
            return o[..., 4:5]
    else:
        gvd_feature_fn = None

    # ── carry helper ─────────────────────────────────────────────────────────

    if args.memory_type == "identity":
        CARRY_DIM = 0
    elif args.memory_type == "lstm":
        CARRY_DIM = 2 * args.memory_hidden_dim
    else:
        CARRY_DIM = args.memory_hidden_dim

    if CARRY_DIM == 0:
        sys.exit("Probe needs recurrent memory (--memory-type rnn/gru/lstm), not identity.")

    # --paper-arch reproduces DiscreteActorCriticRNN: one Dense(H)->ReLU
    # embedding, a GRU(H), single-hidden heads of width H, and no critic
    # LayerNorm.  Otherwise the default LinearProjection (no activation) with
    # (256, 256) heads.
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
        return jnp.zeros((NUM_ACTIONS,), dtype=jnp.float32)

    # ── hyperparameters ──────────────────────────────────────────────────────

    hp = Hyperparameters(
        batch_size=args.batch_size,
        online_buffer_size=args.online_buffer_size,
        target_entropy=args.target_entropy,
        fe_lr=args.fe_lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        lambda_critic_lr=args.critic_lr, alpha_lr=1e-4,
        alpha=args.alpha, autotune_alpha=args.autotune_alpha,
        gamma=args.gamma, tau=args.tau,
        lambda1=args.lambda1, lambda2=args.lambda2,
        c_bar=1.17, rho_bar=1.15, lambda_truncation=17,
        sequence_length=args.sequence_length,
        burn_in_length=args.burn_in_length,
        lambda_coef=args.lambda_coef, fake_onpolicy_loss=False,
        actor_critic=args.actor_critic,
        gvd_coef=args.gvd_coef,
        gvd_lambda1=args.gvd_lambda1,
        gvd_lambda2=args.gvd_lambda2,
        gvd_sf_lr=args.gvd_sf_lr,
        gvd_stop_fe=args.gvd_stop_fe,
        stop_actor_fe=args.stop_actor_fe,
        stop_critic_fe=args.stop_critic_fe,
        ld_center=args.ld_center,
        retrace=args.retrace,
        per_alpha=args.per_alpha,
        per_beta=args.per_beta,
        per_ratio_floor=args.per_ratio_floor,
        per_window=args.per_window,
    )

    _MAX_STEPS = min(int(env_params.max_steps_in_episode), args.eval_max_steps)

    # ── agent / expert ───────────────────────────────────────────────────────

    def _build_agent(seed_val):
        return create_iqlearn_from_env(
            spec, expert_data, buffer_size=1, hp=hp,
            projection=projection_arg,
            memory_type=args.memory_type,
            memory_hidden_dim=args.memory_hidden_dim,
            actor_dims=actor_dims,
            critic_dims=critic_dims,
            lambda1_critic_dims=critic_dims,
            lambda2_critic_dims=critic_dims,
            train_steps=args.train_steps,
            approximate_lambda=args.approximate_lambda,
            use_prev_action=True,
            critic_layer_norm=critic_layer_norm,
            burn_in_from_stored_carry=args.burn_in_from_stored_carry,
            use_gvd=args.gvd, gvd_feature_fn=gvd_feature_fn,
            gvd_sf_dims=(256, 256),
            debug=True, seed=seed_val,
            use_sac=args.use_sac,
        )

    expert_policy = (
        make_pocman_expert(_raw_env, safety_margin=args.expert_safety_margin,
                           chase=not args.expert_no_chase,
                           rollout_depth=args.expert_rollout_depth,
                           epsilon=args.expert_prefill_epsilon)
        if args.expert_prefill_steps > 0 else None
    )

    _transplant_fe, _save_fe = common.make_fe_checkpointer(args.init_fe, args.output_dir)

    tag = ("SAC+LD" if args.approximate_lambda else "SAC") + ("+GVD" if args.gvd else "")
    print(f"Building {tag} agent for PocMan "
          f"(memory={args.memory_type}, hidden={args.memory_hidden_dim}, "
          f"{NUM_PELLETS} pellets)…")
    state, fns, debug_fns = _build_agent(args.seed)

    _eval_kwargs = dict(zero_carry=zero_carry, zero_prev_action=zero_prev_action,
                        max_steps=_MAX_STEPS)
    evaluate = common.make_evaluate(fns, env, env_params, **_eval_kwargs)
    evaluate_critic = (
        common.make_evaluate_critic(debug_fns, env, env_params,
                                    num_actions=NUM_ACTIONS, **_eval_kwargs)
        if (args.critic_greedy_eval
            and getattr(debug_fns, "predict_qpi", None) is not None)
        else None
    )

    # ── rollout collection ───────────────────────────────────────────────────

    @partial(jax.jit, static_argnames=["n_steps", "policy_fn"])
    def collect_rollout(agent_state, key, n_steps, policy_fn=None):
        """Roll out and record (carry, pellet map, player cell) per step.

        ``policy_fn`` (static) replaces the actor as the acting policy — the
        scripted expert, i.e. the distribution an offline agent trained on.  The
        FE is still run every step (that is where the probed carry comes from)
        and the *executed* action is what feeds back as the next prev-action.
        """
        key, rk = jax.random.split(key)
        obs, env_st = env.reset(rk, env_params)
        carry = zero_carry()
        prev_action = zero_prev_action()

        def step_fn(scan_carry, _):
            obs, env_st, carry, prev_action, key = scan_carry
            pellet_alive = jnp.any(env_st.pellet_locations != 0, axis=-1)
            player = jnp.stack([env_st.player_locations.x, env_st.player_locations.y])

            key, sk, ek, eps_key = jax.random.split(key, 4)
            raw, new_carry = fns.predict(
                agent_state, obs, carry, sk, deterministic=False,
                prev_action=prev_action,
            )
            policy_action = jnp.round(raw).astype(jnp.int32)
            random_action = jax.random.randint(eps_key, policy_action.shape, 0, NUM_ACTIONS)
            use_random = jax.random.uniform(eps_key) < args.collect_epsilon
            action = jnp.where(use_random, random_action, policy_action)
            if policy_fn is not None:
                # Scripted policy acts instead (it owns its own exploration).
                scripted, _prob = policy_fn(obs, env_st, eps_key)
                action = jnp.asarray(scripted, dtype=jnp.int32)
            # next prev-action input = the *executed* action (handles the
            # epsilon override transparently)
            new_prev_action = jax.nn.one_hot(action, NUM_ACTIONS)

            next_obs, next_st, _, done, _ = env.step(ek, env_st, action, env_params)
            carry_out = jnp.where(done, zero_carry(), new_carry)
            prev_out = jnp.where(done, zero_prev_action(), new_prev_action)
            return (next_obs, next_st, carry_out, prev_out, key), {
                "carries": new_carry[:CARRY_DIM],
                "pellet_masks": pellet_alive.astype(jnp.float32),
                "players": player.astype(jnp.int32),
                "dones": done.astype(jnp.float32),
            }

        _, data = jax.lax.scan(
            step_fn, (obs, env_st, carry, prev_action, key), length=n_steps)
        return data

    # ── probe helpers ────────────────────────────────────────────────────────

    _train_probe, _train_probe_v, _probe_train_chunk = common.make_probe_trainer(
        optax.adam(args.probe_lr), carry_dim=CARRY_DIM, n_out=NUM_PELLETS,
        hidden=args.probe_hidden_dim, batch_size=args.probe_batch_size,
        tqdm=tqdm, wandb=_wandb)
    _probe_forward_v = jax.vmap(probe_forward)

    _probe_policy = expert_policy if args.probe_rollout_policy != "actor" else None

    def _collect_v(states_b, keys, n_steps):
        return jax.vmap(
            lambda s, k: collect_rollout(s, k, n_steps, policy_fn=_probe_policy)
        )(states_b, keys)

    # ── multi-seed pipeline ──────────────────────────────────────────────────

    CONCURRENT = args.concurrent_seeds or args.num_seeds
    if CONCURRENT < 1:
        sys.exit("--concurrent-seeds must be >= 1")
    if args.num_seeds % CONCURRENT != 0:
        sys.exit(f"--num-seeds ({args.num_seeds}) must be divisible by "
                 f"--concurrent-seeds ({CONCURRENT}).")
    seeds = [args.seed + i for i in range(args.num_seeds)]
    n_groups = args.num_seeds // CONCURRENT
    print(f"{args.num_seeds} seed(s) in {n_groups} group(s) of {CONCURRENT} "
          f"trained concurrently (vmap); {args.rounds} rounds × "
          f"{args.train_steps} steps each.")

    # fns.train has host-side control flow (auto-prefill) and is NOT vmappable;
    # use the vmap-safe split — prefill the buffer once, then run the jittable
    # fns.train_unrolled each round with a per-round zero env_carry.
    PREFILL_STEPS = hp.batch_size * (
        hp.lambda_truncation + hp.sequence_length + hp.burn_in_length)
    _PREFILL_N = max(PREFILL_STEPS, args.expert_prefill_steps)
    _reset_v = jax.jit(jax.vmap(lambda k: env.reset(k, env_params)))
    _prefill_v = jax.jit(
        jax.vmap(lambda s, es, k: fns.prefill_buffer(
            s, env, env_params, es, _PREFILL_N, k, behaviour_fn=expert_policy),
            in_axes=(0, 0, 0)),
        donate_argnums=(0, 1))
    if args.offline:
        # Offline: each round runs train_steps gradient updates on the pre-filled
        # buffer and never touches the environment.  Signature matches
        # train_unrolled so the loop and donation pattern are untouched.
        def _offline_round(s, es, ec, k):
            s, m = fns.update_only(s, args.train_steps, k)
            return s, es, ec, m

        _train_v = jax.jit(jax.vmap(_offline_round, in_axes=(0, 0, 0, 0)),
                           donate_argnums=(0,))
    else:
        _train_v = jax.jit(
            jax.vmap(lambda s, es, ec, k: fns.train_unrolled(s, env, env_params, es, ec, k),
                     in_axes=(0, 0, 0, 0)),
            donate_argnums=(0, 1))

    def _evaluate_v(states_b, keys, n):
        return jax.vmap(lambda s, k: evaluate(s, k, n_episodes=n))(states_b, keys)

    def _evaluate_critic_v(states_b, keys, n):
        return jax.vmap(lambda s, k: evaluate_critic(s, k, n_episodes=n))(states_b, keys)

    return_hist = {gi: [] for gi in range(args.num_seeds)}

    # ── periodic probe-eval (GPU compute → host, then pure-CPU rendering) ────

    def _probe_eval_compute(states_b, gidxs, rnd):
        if args.save_fe_every_eval:
            for j, gi in enumerate(gidxs):
                _save_fe(common.unstack_state(states_b, j), gi, rnd * args.train_steps)
        ne = args.probe_eval_collect_steps
        ck_tr = jnp.stack([jax.random.key(args.seed + 500000 + rnd + gi) for gi in gidxs])
        ck_te = jnp.stack([jax.random.key(args.seed + 600000 + rnd + gi) for gi in gidxs])
        tr = _collect_v(states_b, ck_tr, ne)
        te = _collect_v(states_b, ck_te, ne)
        ik = jnp.stack([jax.random.key(args.seed + 40000 + rnd + gi) for gi in gidxs])
        tk = jnp.stack([jax.random.key(args.seed + 50000 + rnd + gi) for gi in gidxs])
        params_b = _train_probe_v(jnp.asarray(tr["carries"]),
                                  jnp.asarray(tr["pellet_masks"]),
                                  ik, tk, args.probe_eval_steps)
        te_c = jnp.asarray(te["carries"])
        # Pull predictions (and test data) to host — the only sync point;
        # everything downstream is pure CPU and overlaps the next round.
        preds = np.array(jax.nn.sigmoid(_probe_forward_v(params_b, te_c)))
        te_c = np.array(te_c)
        te_t = np.array(te["pellet_masks"]); te_p = np.array(te["players"])
        te_d = np.array(te["dones"])
        per_seed = [
            common.probe_metrics_visitation(
                preds[j], te_t[j], common.episode_bounds(te_d[j], len(te_c[j])))
            for j in range(len(gidxs))
        ]
        return {"gidxs": list(gidxs), "rnd": rnd, "preds": preds, "te_c": te_c,
                "te_t": te_t, "te_p": te_p, "te_d": te_d, "per_seed": per_seed}

    def _probe_eval_render(host):
        gidxs = host["gidxs"]; rnd = host["rnd"]; step = rnd * args.train_steps
        per_seed = host["per_seed"]
        payload = {"env_interactions": step}
        for k in per_seed[0]:
            payload.update(common.agg([m[k] for m in per_seed], f"probe_eval/agg/{k}"))
            for j, gi in enumerate(gidxs):
                payload[f"seed_{gi}/probe_eval/{k}"] = per_seed[j][k]

        def _p(k, fmt=".3f"):
            return f"{payload[f'probe_eval/agg/{k}/mean']:{fmt}}"

        print(f"  [probe-eval] agg AUROC={_p('auroc')}"
              f"±{payload['probe_eval/agg/auroc/sterr']:.3f}  "
              f"errors/state={_p('errors_per_state', '.1f')}  "
              f"exact={_p('exact_match', '.1%')}  "
              f"bits={_p('bits_per_cell', '.4f')}  "
              f"horizon={_p('horizon_steps', '.0f')}  "
              f"eaten recall={_p('eaten_recall', '.1%')}")
        if _wandb is not None:
            _wandb.log(payload)
        if not args.probe_eval_vis:
            return
        img_payload = {"env_interactions": step}
        for j, gi in enumerate(gidxs):
            vtag = f"{tag} @ {step} steps (seed {gi})"
            vdir = os.path.join(args.output_dir, f"seed{gi}", "probe_eval")
            os.makedirs(vdir, exist_ok=True)
            pj, tc = host["preds"][j], host["te_c"][j]
            tt, tp = host["te_t"][j], host["te_p"][j]
            eb = common.episode_bounds(host["te_d"][j], len(tc))
            ep = _fig_episode(None, tc, tt, tp, eb, 0,
                              os.path.join(vdir, f"episode_r{rnd:04d}.png"),
                              vtag, args.vis_frames, PELLET_CELLS, WALL_GRID,
                              probs_all=pj)
            acc = _fig_accuracy(None, tc, tt,
                                os.path.join(vdir, f"accuracy_r{rnd:04d}.png"),
                                vtag, PELLET_CELLS, WALL_GRID, probs_all=pj)
            ret = _fig_retention(None, tc, tt, eb,
                                 os.path.join(vdir, f"retention_r{rnd:04d}.png"),
                                 vtag, probs_all=pj)
            if _wandb is not None:
                if ep is not None:
                    img_payload[f"seed_{gi}/probe_eval/episode"] = _wandb.Image(ep)
                img_payload[f"seed_{gi}/probe_eval/per_cell_accuracy"] = _wandb.Image(acc)
                if ret is not None:
                    img_payload[f"seed_{gi}/probe_eval/retention"] = _wandb.Image(ret)
        if _wandb is not None and len(img_payload) > 1:
            _wandb.log(img_payload)

    # ── train one group of seeds ─────────────────────────────────────────────

    def run_group(group_idx, gidxs):
        svals = [seeds[gi] for gi in gidxs]
        print(f"\n{'=' * 60}\nGroup {group_idx + 1}/{n_groups}  "
              f"seeds={svals} (idx {gidxs[0]}–{gidxs[-1]})\n{'=' * 60}")
        states = [
            _transplant_fe(state if gi == 0 else _build_agent(seeds[gi])[0])
            for gi in gidxs
        ]
        batched = common.stack_states(states)
        keys = jnp.stack([jax.random.key(sv) for sv in svals])
        keys, reset_keys = common.split_each(keys)
        _obs, env_state = _reset_v(reset_keys)
        keys, prefill_keys = common.split_each(keys)
        print(f"  prefilling {_PREFILL_N} steps/seed"
              f"{' from the scripted expert' if expert_policy else ' (random)'}…")
        batched, env_state = _prefill_v(batched, env_state, prefill_keys)
        # Fresh zero env_carry each round (matches fns.train's per-call reset);
        # reused across rounds, so NOT donated.
        zero_carry_b = jnp.zeros((len(gidxs), CARRY_DIM), dtype=jnp.float32)

        def round_eval(keys, batched):
            keys, eval_keys = common.split_each(keys)
            returns, steps, done = _evaluate_v(batched, eval_keys, 10)
            evals = {"return": np.array(returns), "steps": np.array(steps),
                     "done_frac": np.array(done), "cg": None}
            if evaluate_critic is not None:
                keys, cg_keys = common.split_each(keys)
                cgr, cgs, cgd, qsp, qrg = _evaluate_critic_v(batched, cg_keys, 5)
                evals["cg"] = (np.array(cgr), np.array(cgs), np.array(cgd),
                               np.array(qsp), np.array(qrg))
            return keys, evals

        def on_round(rnd, step, evals, metrics):
            returns = evals["return"]; steps = evals["steps"]
            done = evals["done_frac"]; cg = evals["cg"]
            for j, gi in enumerate(gidxs):
                return_hist[gi].append(float(returns[j]))
            print(f"Round {rnd:4d}/{args.rounds}  "
                  f"return={float(returns.mean()):7.1f}±{float(returns.std()):.1f}  "
                  f"steps={float(steps.mean()):5.1f}  "
                  f"finished={float(done.mean()):.2f}")
            if _wandb is None:
                return
            payload = {"round": rnd, "env_interactions": step}
            payload.update(common.agg(returns, "agg/return"))
            payload.update(common.agg(steps, "agg/steps"))
            payload.update(common.agg(done, "agg/done_frac"))
            if cg is not None:
                payload.update(common.agg(cg[0], "agg/critic_greedy_return"))
                payload.update(common.agg(cg[3], "agg/q_action_spread"))
                payload.update(common.agg(cg[4], "agg/q_action_range"))
            for mk, mv in metrics.items():
                payload.update(common.agg(np.array(mv), f"agg/{mk}"))
            for j, gi in enumerate(gidxs):
                payload[f"seed_{gi}/agent/mean_return"] = float(returns[j])
                payload[f"seed_{gi}/agent/steps"] = float(steps[j])
                payload[f"seed_{gi}/agent/done_frac"] = float(done[j])
                for mk, mv in metrics.items():
                    payload[f"seed_{gi}/agent/{mk}"] = float(np.array(mv)[j])
            _wandb.log(payload)

        batched, keys = common.run_seed_group(
            rounds=args.rounds, train_steps=args.train_steps,
            keys=keys, batched=batched, env_state=env_state,
            zero_carry_b=zero_carry_b, train_v=_train_v,
            round_eval=round_eval, on_round=on_round,
            probe=common.Hooks(lambda b, rnd: _probe_eval_compute(b, gidxs, rnd),
                               _probe_eval_render),
            probe_interval=args.probe_eval_interval,
            tqdm=tqdm, desc=f"Group {group_idx + 1}")
        return batched

    # ── Phase 1 — train (or load) every seed ─────────────────────────────────

    indexed = list(range(args.num_seeds))
    groups = [indexed[g * CONCURRENT:(g + 1) * CONCURRENT] for g in range(n_groups)]
    seed_states = [None] * args.num_seeds
    if not args.skip_train:
        for g, gidxs in enumerate(groups):
            batched = run_group(g, gidxs)
            for j, gi in enumerate(gidxs):
                seed_states[gi] = common.unstack_state(batched, j)
                leaves, treedef = jax.tree.flatten(seed_states[gi])
                with open(_seed_path("agent", gi), "wb") as f:
                    pickle.dump({"leaves": [np.array(l) for l in leaves],
                                 "treedef": treedef}, f)
        print(f"Saved {args.num_seeds} per-seed agents → {args.output_dir}/agent_seed*.pkl")

        W = max(1, args.final_return_window)
        fin_ret = np.array([np.mean(return_hist[gi][-W:]) for gi in range(args.num_seeds)])
        agg_final = common.agg(fin_ret, "final/return_smoothed")
        print(f"\n{'=' * 60}\nAggregated over {args.num_seeds} seed(s) "
              f"(smoothed over final {W} round(s)):\n"
              f"  final/return_smoothed = {agg_final['final/return_smoothed/mean']:.1f} "
              f"± {agg_final['final/return_smoothed/sterr']:.1f} (sterr)\n{'=' * 60}")
        if _wandb is not None:
            _wandb.log(agg_final)
            _wandb.summary.update(agg_final)
    else:
        for gi in range(args.num_seeds):
            print(f"Loading agent ← {_seed_path('agent', gi)}")
            with open(_seed_path("agent", gi), "rb") as f:
                saved = pickle.load(f)
            _, treedef = jax.tree.flatten(state)
            seed_states[gi] = treedef.unflatten([jnp.array(l) for l in saved["leaves"]])

    if args.train_only:
        print("--train-only: skipping collect/probe/visualise.")
        if _wandb is not None:
            _wandb.finish()
        sys.exit(0)

    # ── Phase 2 — collect rollouts, per seed (vmapped) ───────────────────────

    all_states = common.stack_states(seed_states)
    if not args.skip_collect:
        print(f"Collecting {args.collect_steps} train+test steps per seed (vmapped)…")
        ck_tr = jnp.stack([jax.random.key(seeds[gi] + 1000) for gi in range(args.num_seeds)])
        ck_te = jnp.stack([jax.random.key(seeds[gi] + 2000) for gi in range(args.num_seeds)])
        tr = _collect_v(all_states, ck_tr, args.collect_steps)
        te = _collect_v(all_states, ck_te, args.collect_steps)
        arrs = {k: (np.array(tr[k]), np.array(te[k]))
                for k in ("carries", "pellet_masks", "players", "dones")}
        for gi in range(args.num_seeds):
            for which, name in ((0, "dataset"), (1, "test_dataset")):
                c = arrs["carries"][which][gi]
                d = arrs["dones"][which][gi]
                with open(_seed_path(name, gi), "wb") as f:
                    pickle.dump({"carries": c,
                                 "pellet_masks": arrs["pellet_masks"][which][gi],
                                 "players": arrs["players"][which][gi],
                                 "ep_bounds": common.episode_bounds(d, len(c)),
                                 "pellet_cells": PELLET_CELLS,
                                 "wall_grid": WALL_GRID, "tag": tag}, f)
        print(f"  → {args.output_dir}/dataset_seed*.pkl, test_dataset_seed*.pkl")

    def _load(name, gi):
        with open(_seed_path(name, gi), "rb") as f:
            return pickle.load(f)

    tr_ds = [_load("dataset", gi) for gi in range(args.num_seeds)]
    te_ds = [_load("test_dataset", gi) for gi in range(args.num_seeds)]

    # ── Phase 3 — train the probe, per seed (vmapped) ────────────────────────

    if not args.skip_probe:
        print(f"Training probe per seed ({args.probe_steps} steps, vmapped)…")
        c_b = jnp.asarray(np.stack([d["carries"] for d in tr_ds]))
        t_b = jnp.asarray(np.stack([d["pellet_masks"] for d in tr_ds]))
        ik = jnp.stack([jax.random.key(seeds[gi] + 20000) for gi in range(args.num_seeds)])
        tk = jnp.stack([jax.random.key(seeds[gi] + 30000) for gi in range(args.num_seeds)])
        probe_b = _train_probe_v(c_b, t_b, ik, tk, args.probe_steps)
        for gi in range(args.num_seeds):
            leaves, td = jax.tree.flatten(common.unstack_state(probe_b, gi))
            with open(_seed_path("probe", gi), "wb") as f:
                pickle.dump({"leaves": [np.array(l) for l in leaves], "treedef": td}, f)
        print(f"  → {args.output_dir}/probe_seed*.pkl")
    else:
        ref = init_probe_params(jax.random.key(0), CARRY_DIM, NUM_PELLETS,
                                args.probe_hidden_dim)
        _, td = jax.tree.flatten(ref)
        probe_b = common.stack_states([
            td.unflatten([jnp.array(l) for l in _load("probe", gi)["leaves"]])
            for gi in range(args.num_seeds)])

    # ── Phase 4 — final metrics + per-seed visualisation ─────────────────────

    print("Computing final probe metrics per seed (held-out test set)…")
    per_seed = []
    for gi in range(args.num_seeds):
        ds = te_ds[gi]
        probs = np.array(jax.nn.sigmoid(probe_forward(
            common.unstack_state(probe_b, gi), jnp.asarray(ds["carries"]))))
        per_seed.append(common.probe_metrics_visitation(
            probs, ds["pellet_masks"], ds["ep_bounds"]))
    final_payload = {}
    for k in per_seed[0]:
        final_payload.update(common.agg([m[k] for m in per_seed], f"eval/agg/{k}"))
        for gi in range(args.num_seeds):
            final_payload[f"eval/seed_{gi}/{k}"] = per_seed[gi][k]
    print(f"  eval/agg/auroc = {final_payload['eval/agg/auroc/mean']:.3f} "
          f"± {final_payload['eval/agg/auroc/sterr']:.3f} (sterr)")
    print(f"  eval/agg/horizon_steps = {final_payload['eval/agg/horizon_steps/mean']:.0f}")
    print(f"  eval/agg/errors_per_state = {final_payload['eval/agg/errors_per_state/mean']:.2f} "
          f"of {NUM_PELLETS} cells")
    if _wandb is not None:
        _wandb.log(final_payload)
        _wandb.summary.update(final_payload)

    print("Rendering per-seed visualisations…")
    for gi in range(args.num_seeds):
        vdir = os.path.join(args.output_dir, f"seed{gi}")
        os.makedirs(vdir, exist_ok=True)
        pp = common.unstack_state(probe_b, gi)
        ds = te_ds[gi]
        c, t, pl, eb = (ds["carries"], ds["pellet_masks"], ds["players"], ds["ep_bounds"])
        vtag = f"{tag} (seed {gi})"
        for vi in range(min(args.vis_episodes, len(eb) - 1)):
            _fig_episode(pp, c, t, pl, eb, vi,
                         os.path.join(vdir, f"pellet_probe_ep{vi + 1}.png"),
                         vtag, args.vis_frames, PELLET_CELLS, WALL_GRID)
        _fig_accuracy(pp, c, t, os.path.join(vdir, "pellet_probe_accuracy.png"),
                      vtag, PELLET_CELLS, WALL_GRID)
        _fig_retention(pp, c, t, eb,
                       os.path.join(vdir, "pellet_probe_retention.png"), vtag)
        if args.mp4:
            lens = np.diff(eb)
            if len(lens):
                best = int(np.argmax(lens))
                s0, e0 = int(eb[best]), int(eb[best + 1])
                _fig_movie(pp, c[s0:e0], t[s0:e0], pl[s0:e0],
                           os.path.join(vdir, "pellet_probe"), vtag,
                           PELLET_CELLS, WALL_GRID)
    print(f"  → per-seed images under {args.output_dir}/seed*/")
    if _wandb is not None:
        _wandb.finish()
    print("Done.")
    sys.exit(0)

# ════════════════════════════════════════════════════════════════════════════
#  Vis-only: render from saved artefacts (needs only jax + matplotlib)
# ════════════════════════════════════════════════════════════════════════════

print(f"Vis-only mode — loading artefacts from {args.output_dir}")
for gi in range(args.num_seeds):
    with open(os.path.join(args.output_dir, f"test_dataset_seed{gi}.pkl"), "rb") as f:
        ds = pickle.load(f)
    with open(os.path.join(args.output_dir, f"probe_seed{gi}.pkl"), "rb") as f:
        saved = pickle.load(f)
    pp = saved["treedef"].unflatten([jnp.array(l) for l in saved["leaves"]])
    cells, wall_grid = ds["pellet_cells"], ds["wall_grid"]
    c, t, pl, eb = ds["carries"], ds["pellet_masks"], ds["players"], ds["ep_bounds"]
    probs = np.array(jax.nn.sigmoid(probe_forward(pp, jnp.asarray(c))))
    m = common.probe_metrics_visitation(probs, t, eb)
    print(f"seed {gi}: AUROC={m['auroc']:.3f}  overall={m['overall_acc']:.1%}  "
          f"eaten recall={m['eaten_recall']:.1%}  horizon={m['horizon_steps']:.0f}  "
          f"errors/state={m['errors_per_state']:.2f}")
    vdir = os.path.join(args.output_dir, f"seed{gi}")
    os.makedirs(vdir, exist_ok=True)
    vtag = f"{ds.get('tag', 'probe')} (seed {gi})"
    for vi in range(min(args.vis_episodes, len(eb) - 1)):
        _fig_episode(None, c, t, pl, eb, vi,
                     os.path.join(vdir, f"pellet_probe_ep{vi + 1}.png"),
                     vtag, args.vis_frames, cells, wall_grid, probs_all=probs)
    _fig_accuracy(None, c, t, os.path.join(vdir, "pellet_probe_accuracy.png"),
                  vtag, cells, wall_grid, probs_all=probs)
    _fig_retention(None, c, t, eb,
                   os.path.join(vdir, "pellet_probe_retention.png"), vtag,
                   probs_all=probs)
    if args.mp4:
        lens = np.diff(eb)
        if len(lens):
            best = int(np.argmax(lens))
            s0, e0 = int(eb[best]), int(eb[best + 1])
            _fig_movie(None, c[s0:e0], t[s0:e0], pl[s0:e0],
                       os.path.join(vdir, "pellet_probe"), vtag, cells, wall_grid,
                       probs_all=probs[s0:e0])
print("Done (vis-only).")
