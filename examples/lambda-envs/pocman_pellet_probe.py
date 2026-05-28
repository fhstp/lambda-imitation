"""Pellet probe for PocMan: train a probe (memory -> pellet occupancy) and visualise.

Replicates the analysis from Section I.5 / Figure 6 (right) of:
  Allen et al., "Mitigating Partial Observability in Sequential Decision
  Processes via the Lambda Discrepancy" (NeurIPS 2024, arXiv:2407.07333).

Pipeline:
  1. Train a PocMan SAC(+LD) agent for 80 k env steps (default)
  2. Collect rollouts (step-based, auto-reset) saving hidden states and true pellet occupancy
  3. Train a 3-layer MLP probe: hidden state -> pellet occupancy (BCE)
  4. Visualise probe predictions vs ground truth on the maze grid

All artefacts are saved under --output-dir so individual phases can be
skipped on subsequent runs (--skip-train, --skip-collect, --skip-probe).
Use the same CLI flags when re-loading to ensure consistent architecture.

Vis-only mode (--vis-only) needs only dataset.pkl + probe.pkl — no
lambda-imitation or lambda-envs install required (just jax + matplotlib).

Usage:
    python pocman_pellet_probe.py                          # full pipeline
    python pocman_pellet_probe.py --skip-train             # reuse saved agent
    python pocman_pellet_probe.py --no-approximate-lambda  # SAC-only baseline
    python pocman_pellet_probe.py --vis-only --mp4         # just render (portable)
"""

import argparse
import os
import pickle
import sys
from functools import partial

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="PocMan pellet probe visualisation.")

g = parser.add_argument_group("agent training")
g.add_argument("--rounds", type=int, default=8, help="training rounds (default 8)")
g.add_argument("--train-steps", type=int, default=10_000, help="env steps per round (default 10 000)")
g.add_argument("--seed", type=int, default=42)
g.add_argument("--memory-type", choices=("identity", "rnn", "gru", "lstm"), default="gru")
g.add_argument("--memory-hidden-dim", type=int, default=750)
g.add_argument("--projection-dim", type=int, default=128)
g.add_argument("--approximate-lambda", dest="approximate_lambda", action="store_true")
g.add_argument("--no-approximate-lambda", dest="approximate_lambda", action="store_false")
parser.set_defaults(approximate_lambda=True)
g.add_argument("--batch-size", type=int, default=512)
g.add_argument("--sequence-length", type=int, default=20)
g.add_argument("--burn-in-length", type=int, default=32)
g.add_argument("--online-buffer-size", type=int, default=200_000)

g = parser.add_argument_group("data collection")
g.add_argument("--collect-steps", type=int, default=100_000, help="total env steps to collect (default 100 000)")

g = parser.add_argument_group("probe training")
g.add_argument("--probe-steps", type=int, default=500_000, help="SGD steps (default 500 k)")
g.add_argument("--probe-lr", type=float, default=1e-4)
g.add_argument("--probe-hidden-dim", type=int, default=1024)
g.add_argument("--probe-batch-size", type=int, default=32)

g = parser.add_argument_group("I/O & visualisation")
g.add_argument("--output-dir", default="./pocman_probe_output")
g.add_argument("--skip-train", action="store_true", help="load agent from output-dir")
g.add_argument("--skip-collect", action="store_true", help="load dataset from output-dir")
g.add_argument("--skip-probe", action="store_true", help="load probe from output-dir")
g.add_argument("--vis-only", action="store_true",
               help="portable mode: load dataset.pkl + probe.pkl, render only "
                    "(no lambda-imitation / lambda-envs needed)")
g.add_argument("--mp4", action="store_true",
               help="save an mp4 (or gif fallback) of the longest collected episode")
g.add_argument("--vis-episodes", type=int, default=3, help="episodes to visualise")
g.add_argument("--vis-frames", type=int, default=8, help="frames per episode")

args, _ = parser.parse_known_args()

# ── always-needed imports ────────────────────────────────────────────────────

import jax
import jax.numpy as jnp
import numpy as np

# ── probe MLP (pure JAX — no external deps) ──────────────────────────────────


def init_probe_params(key, carry_dim, n_out, hidden=1024):
    k1, k2, k3 = jax.random.split(key, 3)

    def layer(k, din, dout):
        return {
            "w": jax.random.normal(k, (din, dout)) * (2.0 / din) ** 0.5,
            "b": jnp.zeros(dout),
        }

    return {
        "l1": layer(k1, carry_dim, hidden),
        "l2": layer(k2, hidden, hidden),
        "l3": layer(k3, hidden, n_out),
    }


def probe_forward(params, x):
    x = jax.nn.relu(x @ params["l1"]["w"] + params["l1"]["b"])
    x = jax.nn.relu(x @ params["l2"]["w"] + params["l2"]["b"])
    return x @ params["l3"]["w"] + params["l3"]["b"]


# ── output paths ─────────────────────────────────────────────────────────────

os.makedirs(args.output_dir, exist_ok=True)
agent_path = os.path.join(args.output_dir, "agent.pkl")
dataset_path = os.path.join(args.output_dir, "dataset.pkl")
probe_path = os.path.join(args.output_dir, "probe.pkl")


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
    from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax

    # ── env setup ────────────────────────────────────────────────────────────

    class _LambdaEnvAdapter:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def get_obs(self, state, params=None):
            return self._wrapped.get_obs(state)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    env = _LambdaEnvAdapter(PocMan())
    env_params = env.default_params
    spec = env_spec_from_gymnax(env, env_params)

    expert_data = {
        "observations": jnp.zeros((1, *spec.obs_shape), dtype=jnp.float32),
        "actions": jnp.zeros((1, 1), dtype=jnp.float32),
    }

    # ── maze geometry ────────────────────────────────────────────────────────

    MAZE_ROWS = len(SMALLER_GAME_MAP)
    MAZE_COLS = len(SMALLER_GAME_MAP[0])
    WALL_GRID = np.array([[c == "X" for c in row] for row in SMALLER_GAME_MAP])

    # ── carry helper ─────────────────────────────────────────────────────────

    if args.memory_type == "identity":
        CARRY_DIM = 0
    elif args.memory_type == "lstm":
        CARRY_DIM = 2 * args.memory_hidden_dim
    else:
        CARRY_DIM = args.memory_hidden_dim

    if CARRY_DIM == 0:
        sys.exit("Probe needs recurrent memory (--memory-type rnn/gru/lstm), not identity.")

    projection_dim = args.projection_dim if args.projection_dim > 0 else None

    def zero_carry():
        return jnp.zeros((CARRY_DIM,), dtype=jnp.float32)

    # ── hyperparameters ──────────────────────────────────────────────────────

    hp = Hyperparameters(
        online_batch_size=128,
        online_buffer_size=args.online_buffer_size,
        target_entropy=0.0,
        fe_lr=7e-5, actor_lr=6e-5, critic_lr=2e-4,
        lambda_critic_lr=1.8e-4, alpha_lr=1e-4,
        alpha=0.1, autotune_alpha=False,
        batch_size=args.batch_size, gamma=0.95, tau=0.006,
        lambda1=0.05, lambda2=0.8,
        c_bar=1.17, rho_bar=1.15, lambda_truncation=17,
        sequence_length=args.sequence_length,
        burn_in_length=args.burn_in_length,
        lambda_coef=1.0, fake_onpolicy_loss=False,
    )

    _MAX_STEPS = int(env_params.max_steps_in_episode)

    def _build_agent(seed_val):
        return create_iqlearn_from_env(
            spec, expert_data, buffer_size=1, hp=hp,
            projection_dim=projection_dim,
            memory_type=args.memory_type,
            memory_hidden_dim=args.memory_hidden_dim,
            critic_dims=(128,),
            lambda1_critic_dims=(128,),
            lambda2_critic_dims=(128,),
            train_steps=args.train_steps,
            approximate_lambda=args.approximate_lambda,
            debug=True, seed=seed_val,
        )

    # ── evaluation helper ────────────────────────────────────────────────────

    def _make_evaluate(fns):
        @partial(jax.jit, static_argnames=["n_episodes"])
        def _evaluate(agent_state, rng_key, n_episodes=10):
            def run_episode(key):
                key, rk = jax.random.split(key)
                obs, env_st = env.reset(rk, env_params)
                carry = zero_carry()
                def step_fn(s, _):
                    obs, env_st, carry, key, ret, done = s
                    key, sk, ek = jax.random.split(key, 3)
                    raw, nc = fns.predict(agent_state, obs, carry, sk, deterministic=True)
                    action = jnp.round(raw).astype(jnp.int32)
                    nobs, nst, rew, d, _ = env.step(ek, env_st, action, env_params)
                    ret = ret + rew * (1.0 - done)
                    done = jnp.maximum(done, d.astype(jnp.float32))
                    return (nobs, nst, nc, key, ret, done), None
                init = (obs, env_st, carry, key, jnp.float32(0.0), jnp.float32(0.0))
                (_, _, _, _, ep_ret, _), _ = jax.lax.scan(step_fn, init, length=_MAX_STEPS)
                return ep_ret
            keys = jax.random.split(rng_key, n_episodes)
            return jnp.mean(jax.vmap(run_episode)(keys))
        return _evaluate

    # ── Phase 1 — Train / load agent ─────────────────────────────────────────

    tag = "SAC+LD" if args.approximate_lambda else "SAC"
    print(f"Building {tag} agent (memory={args.memory_type}, hidden={args.memory_hidden_dim})…")
    state, fns, debug_fns = _build_agent(args.seed)
    evaluate = _make_evaluate(fns)

    if not args.skip_train:
        key = jax.random.key(args.seed)
        key, reset_key = jax.random.split(key)
        _, env_state = env.reset(reset_key, env_params)
        total = args.rounds * args.train_steps
        print(f"Training for {args.rounds} × {args.train_steps} = {total} steps…")

        for rnd in tqdm(range(1, args.rounds + 1), desc="Training"):
            key, train_key = jax.random.split(key)
            state, env_state, metrics = fns.train(
                state, env, env_params, env_state, train_key
            )
            key, eval_key = jax.random.split(key)
            mr = float(evaluate(state, eval_key, n_episodes=10))
            print(
                f"  round {rnd:3d}/{args.rounds}  return={mr:7.1f}  "
                f"critic_loss={float(metrics.get('critic_loss', jnp.nan)):.4f}"
            )

        print(f"Saving agent → {agent_path}")
        leaves, treedef = jax.tree.flatten(state)
        with open(agent_path, "wb") as f:
            pickle.dump(
                {"leaves": [np.array(l) for l in leaves], "treedef": treedef}, f
            )
    else:
        print(f"Loading agent ← {agent_path}")
        with open(agent_path, "rb") as f:
            saved = pickle.load(f)
        _, treedef = jax.tree.flatten(state)
        state = treedef.unflatten([jnp.array(l) for l in saved["leaves"]])

    # ── Phase 2 — Collect rollouts ───────────────────────────────────────────

    if not args.skip_collect:
        dummy_obs, dummy_st = env.reset(jax.random.key(0), env_params)
        init_pellet_locs = np.array(dummy_st.pellet_locations)
        num_pellets = init_pellet_locs.shape[0]
        print(f"Maze {MAZE_ROWS}×{MAZE_COLS}, {num_pellets} pellet slots.")
        print(f"Collecting {args.collect_steps} steps (auto-reset)…")

        @partial(jax.jit, static_argnames=["n_steps"])
        def collect_rollout(agent_state, key, n_steps):
            key, rk = jax.random.split(key)
            obs, env_st = env.reset(rk, env_params)
            carry = zero_carry()

            def step_fn(scan_carry, _):
                obs, env_st, carry, key = scan_carry
                pellet_alive = jnp.any(env_st.pellet_locations != 0, axis=-1)
                player_row = env_st.player_locations.x
                player_col = env_st.player_locations.y

                key, sk, ek = jax.random.split(key, 3)
                raw, new_carry = fns.predict(
                    agent_state, obs, carry, sk, deterministic=True
                )
                action = jnp.round(raw).astype(jnp.int32)

                next_obs, next_st, _, done, _ = env.step(
                    ek, env_st, action, env_params
                )
                carry_out = jnp.where(done, zero_carry(), new_carry)
                return (next_obs, next_st, carry_out, key), {
                    "carries": new_carry,
                    "pellet_masks": pellet_alive.astype(jnp.float32),
                    "player_rows": player_row,
                    "player_cols": player_col,
                    "dones": done.astype(jnp.float32),
                }

            _, data = jax.lax.scan(step_fn, (obs, env_st, carry, key), length=n_steps)
            return data

        key = jax.random.key(args.seed + 1000)
        data = collect_rollout(state, key, args.collect_steps)

        carries = np.array(data["carries"])
        pellet_masks = np.array(data["pellet_masks"])
        player_rows = np.array(data["player_rows"])
        player_cols = np.array(data["player_cols"])
        dones = np.array(data["dones"])

        done_idxs = np.where(dones > 0.5)[0]
        ep_starts = np.concatenate([[0], done_idxs + 1])
        ep_starts = ep_starts[ep_starts < len(carries)]
        ep_bounds = np.concatenate([ep_starts, [len(carries)]])

        print(f"Collected {len(carries)} steps, {len(ep_starts)} episodes.  Saving → {dataset_path}")
        with open(dataset_path, "wb") as f:
            pickle.dump(
                {
                    "carries": carries,
                    "pellet_masks": pellet_masks,
                    "player_rows": player_rows,
                    "player_cols": player_cols,
                    "ep_bounds": ep_bounds,
                    "init_pellet_locs": init_pellet_locs,
                    "wall_grid": WALL_GRID,
                    "tag": tag,
                },
                f,
            )
    else:
        print(f"Loading dataset ← {dataset_path}")
        with open(dataset_path, "rb") as f:
            ds = pickle.load(f)
        carries = ds["carries"]
        pellet_masks = ds["pellet_masks"]
        player_rows = ds["player_rows"]
        player_cols = ds["player_cols"]
        ep_bounds = ds["ep_bounds"]
        init_pellet_locs = ds["init_pellet_locs"]
        WALL_GRID = ds.get("wall_grid", WALL_GRID)
        tag = ds.get("tag", tag)
        num_pellets = init_pellet_locs.shape[0]
        print(f"Loaded {len(carries)} timesteps, {num_pellets} pellet slots.")

    # ── Phase 3 — Train probe ────────────────────────────────────────────────

    if not args.skip_probe:
        print(
            f"Training probe ({args.probe_steps} steps, "
            f"hidden={args.probe_hidden_dim}, lr={args.probe_lr})…"
        )
        probe_params = init_probe_params(
            jax.random.key(args.seed + 2000),
            CARRY_DIM, num_pellets, args.probe_hidden_dim,
        )
        opt = optax.adam(args.probe_lr)
        opt_state = opt.init(probe_params)

        c_jnp = jnp.array(carries)
        t_jnp = jnp.array(pellet_masks)
        n_samples = c_jnp.shape[0]

        @partial(jax.jit, static_argnames=["n_steps", "batch_size"])
        def probe_train_chunk(params, opt_state, key, c_data, t_data, n_steps, batch_size):
            n = c_data.shape[0]
            def body(carry, _):
                params, opt_state, key = carry
                key, bk = jax.random.split(key)
                idx = jax.random.randint(bk, (batch_size,), 0, n)
                def loss_fn(p):
                    logits = probe_forward(p, c_data[idx])
                    return optax.sigmoid_binary_cross_entropy(logits, t_data[idx]).mean()
                loss, grads = jax.value_and_grad(loss_fn)(params)
                updates, new_os = opt.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), new_os, key), loss
            (params, opt_state, key), losses = jax.lax.scan(body, (params, opt_state, key), length=n_steps)
            return params, opt_state, key, losses[-1]

        _CHUNK = 50_000
        n_chunks = args.probe_steps // _CHUNK
        remainder = args.probe_steps % _CHUNK
        key = jax.random.key(args.seed + 3000)
        steps_done = 0

        for _ in tqdm(range(n_chunks), desc="Probe"):
            probe_params, opt_state, key, loss_val = probe_train_chunk(
                probe_params, opt_state, key, c_jnp, t_jnp, _CHUNK, args.probe_batch_size
            )
            steps_done += _CHUNK
            print(f"  step {steps_done:7d}  bce={float(loss_val):.6f}")

        if remainder > 0:
            probe_params, opt_state, key, loss_val = probe_train_chunk(
                probe_params, opt_state, key, c_jnp, t_jnp, remainder, args.probe_batch_size
            )
            steps_done += remainder
            print(f"  step {steps_done:7d}  bce={float(loss_val):.6f}")

        print(f"Saving probe → {probe_path}")
        leaves, td = jax.tree.flatten(probe_params)
        with open(probe_path, "wb") as f:
            pickle.dump({"leaves": [np.array(l) for l in leaves], "treedef": td}, f)
    else:
        print(f"Loading probe ← {probe_path}")
        with open(probe_path, "rb") as f:
            saved = pickle.load(f)
        ref = init_probe_params(
            jax.random.key(0), CARRY_DIM, num_pellets, args.probe_hidden_dim
        )
        _, td = jax.tree.flatten(ref)
        probe_params = td.unflatten([jnp.array(l) for l in saved["leaves"]])


# ════════════════════════════════════════════════════════════════════════════
#  Vis-only: load dataset + probe (no lambda-imitation needed)
# ════════════════════════════════════════════════════════════════════════════

if args.vis_only:
    print(f"Vis-only mode — loading artefacts from {args.output_dir}")

    with open(dataset_path, "rb") as f:
        ds = pickle.load(f)
    carries = ds["carries"]
    pellet_masks = ds["pellet_masks"]
    player_rows = ds["player_rows"]
    player_cols = ds["player_cols"]
    ep_bounds = ds["ep_bounds"]
    init_pellet_locs = ds["init_pellet_locs"]
    WALL_GRID = ds["wall_grid"]
    tag = ds.get("tag", "SAC+LD")
    num_pellets = init_pellet_locs.shape[0]
    CARRY_DIM = carries.shape[1]
    print(f"  dataset: {len(carries)} timesteps, {num_pellets} pellets, "
          f"{len(ep_bounds)-1} episodes")

    with open(probe_path, "rb") as f:
        saved = pickle.load(f)
    probe_params = saved["treedef"].unflatten(
        [jnp.array(l) for l in saved["leaves"]]
    )
    print(f"  probe loaded")


# ════════════════════════════════════════════════════════════════════════════
#  Phase 4 — Visualise (shared rendering code)
# ════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAZE_ROWS, MAZE_COLS = WALL_GRID.shape
pellet_row = init_pellet_locs[:, 1]
pellet_col = init_pellet_locs[:, 0]

WALL_COLOR = np.array([0.12, 0.12, 0.18])
EMPTY_COLOR = np.array([0.28, 0.28, 0.30])
PELLET_COLOR = np.array([0.10, 0.88, 0.18])
PLAYER_COLOR = np.array([1.00, 0.90, 0.10])


def to_grid(values):
    """Map per-pellet values to a (rows, cols) grid.  NaN = not a pellet slot."""
    g = np.full((MAZE_ROWS, MAZE_COLS), np.nan)
    for i in range(len(values)):
        r, c = int(pellet_row[i]), int(pellet_col[i])
        if 0 <= r < MAZE_ROWS and 0 <= c < MAZE_COLS:
            g[r, c] = values[i]
    return g


def render(ax, pellet_grid, pr, pc, title=""):
    img = np.tile(WALL_COLOR, (MAZE_ROWS, MAZE_COLS, 1))
    for r in range(MAZE_ROWS):
        for c in range(MAZE_COLS):
            if not WALL_GRID[r, c]:
                v = pellet_grid[r, c]
                if np.isnan(v):
                    img[r, c] = EMPTY_COLOR
                else:
                    img[r, c] = EMPTY_COLOR * (1 - v) + PELLET_COLOR * v
    img[pr, pc] = PLAYER_COLOR
    ax.imshow(img, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def _vis_stored_episode(ve_idx, probe_params):
    """Render one stored episode (truth vs probe) as a multi-frame PNG."""
    start, end = ep_bounds[ve_idx], ep_bounds[ve_idx + 1]
    ep_c = carries[start:end]
    ep_m = pellet_masks[start:end]
    ep_r = player_rows[start:end]
    ep_col = player_cols[start:end]
    ep_len = end - start

    preds = np.array(jax.nn.sigmoid(probe_forward(probe_params, jnp.array(ep_c))))

    nf = min(args.vis_frames, ep_len)
    idxs = np.linspace(0, ep_len - 1, nf, dtype=int)

    fig, axes = plt.subplots(2, nf, figsize=(2.4 * nf, 6))
    if nf == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(
        f"Pellet Probe — {tag}  episode {ve_idx + 1}  (len={ep_len})",
        fontsize=10,
    )
    for j, idx in enumerate(idxs):
        render(axes[0, j], to_grid(ep_m[idx]), ep_r[idx], ep_col[idx], f"True  t={idx}")
        render(axes[1, j], to_grid(preds[idx]), ep_r[idx], ep_col[idx], f"Pred  t={idx}")
    axes[0, 0].set_ylabel("Ground Truth", fontsize=9)
    axes[1, 0].set_ylabel("Probe", fontsize=9)
    fig.tight_layout()
    p = os.path.join(args.output_dir, f"pellet_probe_ep{ve_idx + 1}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ── episode PNGs ─────────────────────────────────────────────────────────────

if args.vis_only:
    n_eps = min(args.vis_episodes, len(ep_bounds) - 1)
    print(f"Generating {n_eps} episode visualisation(s) from stored data…")
    for ve in range(n_eps):
        p = _vis_stored_episode(ve, probe_params)
        print(f"  → {p}")
else:
    probe_jit = jax.jit(partial(probe_forward, probe_params))
    print(f"Generating {args.vis_episodes} episode visualisation(s)…")

    key = jax.random.key(args.seed + 5000)
    for ve in range(args.vis_episodes):
        key, rk = jax.random.split(key)
        obs, env_st = env.reset(rk, env_params)
        carry = zero_carry()

        true_grids, pred_grids, prs, pcs = [], [], [], []
        for _ in range(_MAX_STEPS):
            alive = np.array(
                jnp.any(env_st.pellet_locations != 0, axis=-1), dtype=np.float32
            )
            key, sk, ek = jax.random.split(key, 3)
            raw, new_carry = fns.predict(
                state, obs, carry, sk, deterministic=True
            )
            action = jnp.round(raw).astype(jnp.int32)

            pred = np.array(jax.nn.sigmoid(probe_jit(new_carry)))
            true_grids.append(to_grid(alive))
            pred_grids.append(to_grid(pred))
            prs.append(int(env_st.player_locations.x))
            pcs.append(int(env_st.player_locations.y))

            next_obs, next_st, _, done, _ = env.step(
                ek, env_st, action, env_params
            )
            obs, env_st = next_obs, next_st
            carry = jnp.where(done, zero_carry(), new_carry)
            if done:
                break

        ep_len = len(true_grids)
        nf = min(args.vis_frames, ep_len)
        idxs = np.linspace(0, ep_len - 1, nf, dtype=int)

        fig, axes = plt.subplots(2, nf, figsize=(2.4 * nf, 6))
        if nf == 1:
            axes = axes.reshape(2, 1)
        fig.suptitle(
            f"Pellet Probe — {tag}  episode {ve + 1}  (len={ep_len})",
            fontsize=10,
        )
        for j, idx in enumerate(idxs):
            render(axes[0, j], true_grids[idx], prs[idx], pcs[idx], f"True  t={idx}")
            render(axes[1, j], pred_grids[idx], prs[idx], pcs[idx], f"Pred  t={idx}")
        axes[0, 0].set_ylabel("Ground Truth", fontsize=9)
        axes[1, 0].set_ylabel("Probe", fontsize=9)
        fig.tight_layout()
        p = os.path.join(args.output_dir, f"pellet_probe_ep{ve + 1}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  → {p}")

# ── accuracy summary ─────────────────────────────────────────────────────────

print("Computing overall probe accuracy…")
all_logits = probe_forward(probe_params, jnp.array(carries))
preds_bin = (jax.nn.sigmoid(all_logits) > 0.5).astype(jnp.float32)
targets = jnp.array(pellet_masks)
acc = float(jnp.mean(preds_bin == targets))
per_pellet = np.array(jnp.mean(preds_bin == targets, axis=0))
print(f"Overall accuracy: {acc:.1%}")

fig, ax = plt.subplots(figsize=(5.5, 6.5))
g = to_grid(per_pellet)
img = np.tile(WALL_COLOR, (MAZE_ROWS, MAZE_COLS, 1))
for r in range(MAZE_ROWS):
    for c in range(MAZE_COLS):
        if not WALL_GRID[r, c]:
            v = g[r, c]
            if np.isnan(v):
                img[r, c] = EMPTY_COLOR
            else:
                img[r, c] = [1.0 - v, v, 0.15]
ax.imshow(img, interpolation="nearest", aspect="equal")
ax.set_title(f"Per-Cell Probe Accuracy — {tag}\nOverall: {acc:.1%}", fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
p = os.path.join(args.output_dir, "pellet_probe_accuracy.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"  → {p}")

# ── mp4 / gif ────────────────────────────────────────────────────────────────

if args.mp4:
    from matplotlib.animation import FuncAnimation

    ep_lens = np.diff(ep_bounds)
    best = int(np.argmax(ep_lens))
    start, end = int(ep_bounds[best]), int(ep_bounds[best + 1])
    ep_c = carries[start:end]
    ep_m = pellet_masks[start:end]
    ep_r = player_rows[start:end]
    ep_col = player_cols[start:end]
    ep_len = end - start

    ep_preds = np.array(
        jax.nn.sigmoid(probe_forward(probe_params, jnp.array(ep_c)))
    )

    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(10, 6))

    def _update(frame):
        ax_t.clear()
        ax_p.clear()
        render(ax_t, to_grid(ep_m[frame]), ep_r[frame], ep_col[frame], "Ground Truth")
        render(ax_p, to_grid(ep_preds[frame]), ep_r[frame], ep_col[frame], "Probe Prediction")
        fig.suptitle(f"Pellet Probe — {tag}   t = {frame}/{ep_len}", fontsize=11)

    print(f"Rendering mp4 for episode {best + 1} ({ep_len} frames)…")
    anim = FuncAnimation(fig, _update, frames=ep_len, interval=100)

    mp4_path = os.path.join(args.output_dir, "pellet_probe.mp4")
    try:
        anim.save(mp4_path, writer="ffmpeg", fps=10)
        print(f"  → {mp4_path}")
    except Exception:
        gif_path = os.path.join(args.output_dir, "pellet_probe.gif")
        anim.save(gif_path, writer="pillow", fps=10)
        print(f"  ffmpeg unavailable, saved gif → {gif_path}")
    plt.close(fig)

print("Done.")
