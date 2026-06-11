"""Board probe for Battleship: train a probe (memory -> ship occupancy) and visualise.

Battleship analogue of ``pocman_pellet_probe.py``.  The agent only ever observes
a single bit per step — did the last shot hit — so to localise ships at all its
recurrent memory must integrate the *action–observation history*.  The previous
action therefore reaches the feature extractor via ``use_prev_action=True``
(packed-carry prev-action input; the legal-action mask stays out of the
network, used only to mask the policy).  A probe then tests whether the memory
has reconstructed the hidden board.

Pipeline:
  1. Train a Battleship SAC(+LD) agent (action-masked)
  2. Collect rollouts (auto-reset) saving hidden states + the true ship board
     and the hits/misses fired so far
  3. Train a 3-layer MLP probe: hidden state -> ship occupancy (BCE)
  4. Visualise probe predictions vs the true board, with the agent's shots
     overlaid, on the rows×cols grid

All artefacts are saved under --output-dir so individual phases can be skipped
on subsequent runs (--skip-train, --skip-collect, --skip-probe).  Vis-only mode
(--vis-only) needs only dataset.pkl + probe.pkl (just jax + matplotlib).

Usage:
    python battleship_board_probe.py                          # full pipeline
    python battleship_board_probe.py --rows 5 --cols 5        # smaller board
    python battleship_board_probe.py --skip-train             # reuse saved agent
    python battleship_board_probe.py --no-approximate-lambda  # SAC-only baseline
    python battleship_board_probe.py --vis-only --mp4         # just render (portable)
"""

import argparse
import os
import pickle
import sys
from functools import partial

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Battleship board probe visualisation.")

g = parser.add_argument_group("environment")
g.add_argument("--rows", type=int, default=10, help="board rows (default 10)")
g.add_argument("--cols", type=int, default=10, help="board cols (default 10)")
g.add_argument("--dense-reward", dest="dense_reward", action="store_true",
               help="reward every hit (default: sparse terminal reward)")
parser.set_defaults(dense_reward=False)

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
g.add_argument("--gvd", dest="gvd", action="store_true",
               help="enable the GVD successor-feature branches (reward-free "
                    "memory pressure; an agent trained with --gvd must be "
                    "reloaded with --gvd)")
g.add_argument("--no-gvd", dest="gvd", action="store_false")
parser.set_defaults(gvd=False)
g.add_argument("--gvd-coef", type=float, default=1.0, help="GVD discrepancy coefficient (default 1.0)")
g.add_argument("--gvd-features", type=int, default=16,
               help="random-projection width of the GVD feature map; total dim "
                    "is 1 (hit bit) + N (default 16)")
g.add_argument("--gvd-lambda1", type=float, default=0.0)
g.add_argument("--gvd-lambda2", type=float, default=1.0)
g.add_argument("--gvd-sf-lr", type=float, default=1.8e-4)
g.add_argument("--batch-size", type=int, default=512)
g.add_argument("--sequence-length", type=int, default=20)
g.add_argument("--burn-in-length", type=int, default=32)
g.add_argument("--burn-in-from-stored-carry", dest="burn_in_from_stored_carry",
               action="store_true",
               help="store the online carry per transition and initialise the "
                    "training burn-in from it instead of zeros (R2D2 "
                    "stored-state; enables shorter --burn-in-length; costs "
                    "carry_dim x buffer_size x 4 bytes extra)")
g.add_argument("--no-burn-in-from-stored-carry", dest="burn_in_from_stored_carry",
               action="store_false")
parser.set_defaults(burn_in_from_stored_carry=False)
g.add_argument("--online-buffer-size", type=int, default=200_000)

g = parser.add_argument_group("data collection")
g.add_argument("--collect-steps", type=int, default=100_000, help="total env steps to collect (default 100 000)")
g.add_argument("--collect-epsilon", type=float, default=0.1, help="epsilon-greedy rate during collection (default 0.1)")

g = parser.add_argument_group("probe training")
g.add_argument("--probe-steps", type=int, default=500_000, help="SGD steps (default 500 k)")
g.add_argument("--probe-lr", type=float, default=1e-4)
g.add_argument("--probe-hidden-dim", type=int, default=1024)
g.add_argument("--probe-batch-size", type=int, default=32)

g = parser.add_argument_group("I/O & visualisation")
g.add_argument("--output-dir", default="./battleship_probe_output")
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
test_dataset_path = os.path.join(args.output_dir, "test_dataset.pkl")
probe_path = os.path.join(args.output_dir, "probe.pkl")


# ════════════════════════════════════════════════════════════════════════════
#  Full pipeline (phases 1-3) — skipped entirely in --vis-only
# ════════════════════════════════════════════════════════════════════════════

if not args.vis_only:
    import optax
    from tqdm.rich import tqdm

    try:
        from lambda_envs.envs.battleship import Battleship
    except ImportError:
        sys.exit("lambda-envs required.  pip install lambda-envs")

    from lambda_imitation.iqlearn import Hyperparameters
    from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax

    # ── env setup ──────────────────────────────────────────────────────────────
    #
    # The previous action reaches the feature extractor via
    # ``use_prev_action=True`` (its one-hot lives in the carry tail; see
    # RecurrentFeatureExtractor), so the env is used unwrapped.  The memory
    # then sees *which* cell was fired and *whether* it hit, which is exactly
    # what it needs to reconstruct the hidden board.

    env = Battleship(rows=args.rows, cols=args.cols, dense_reward=args.dense_reward)
    env_params = env.default_params
    spec = env_spec_from_gymnax(env, env_params)
    N = args.rows * args.cols  # number of actions / board cells
    # obs layout: [last_hit_miss(1) | mask(N)]
    obs_fn = lambda o: o[..., 0:1]
    mask_fn = lambda o: o[..., 1:1 + N]

    expert_data = {
        "observations": jnp.zeros((1, *spec.obs_shape), dtype=jnp.float32),
        "actions": jnp.zeros((1, 1), dtype=jnp.float32),
    }

    # GVD feature map φ = [hit bit | P·obs] over the FULL raw observation —
    # identical construction to battleship_sac_mc.py (fixed jax.random.key(0)
    # projection, independent of --seed) so agents trained there probe the
    # same way here.
    if args.gvd:
        _GVD_P = jax.random.normal(
            jax.random.key(0), (1 + N, args.gvd_features)
        ) / jnp.sqrt(1 + N)

        def gvd_feature_fn(o):
            return jnp.concatenate([o[..., :1], o @ _GVD_P], axis=-1)
    else:
        gvd_feature_fn = None

    # ── carry helper ─────────────────────────────────────────────────────────
    #
    # CARRY_DIM is the *memory* part (the probe input, as before).  The agent
    # carry additionally holds the prev-action one-hot in its tail
    # (use_prev_action=True), so the full carry is CARRY_DIM + N wide.

    if args.memory_type == "identity":
        CARRY_DIM = 0
    elif args.memory_type == "lstm":
        CARRY_DIM = 2 * args.memory_hidden_dim
    else:
        CARRY_DIM = args.memory_hidden_dim

    if CARRY_DIM == 0:
        sys.exit("Probe needs recurrent memory (--memory-type rnn/gru/lstm), not identity.")

    AGENT_CARRY_DIM = CARRY_DIM + N  # memory + prev-action tail

    projection_dim = args.projection_dim if args.projection_dim > 0 else None

    def zero_carry():
        return jnp.zeros((AGENT_CARRY_DIM,), dtype=jnp.float32)

    # ── hyperparameters ──────────────────────────────────────────────────────

    hp = Hyperparameters(
        online_batch_size=128,
        online_buffer_size=args.online_buffer_size,
        target_entropy=0.0,
        fe_lr=7e-5, actor_lr=6e-5, critic_lr=2e-4,
        lambda_critic_lr=1.8e-4, alpha_lr=1e-4,
        alpha=0.1, autotune_alpha=False,
        batch_size=args.batch_size, gamma=0.99, tau=0.006,
        lambda1=0.05, lambda2=0.8,
        c_bar=1.17, rho_bar=1.15, lambda_truncation=17,
        sequence_length=args.sequence_length,
        burn_in_length=args.burn_in_length,
        lambda_coef=1.0, fake_onpolicy_loss=False,
        gvd_coef=args.gvd_coef,
        gvd_lambda1=args.gvd_lambda1,
        gvd_lambda2=args.gvd_lambda2,
        gvd_sf_lr=args.gvd_sf_lr,
    )

    # Battleship episodes end after at most rows*cols shots (legal-action
    # mask exhausts the board) — scan only that far, not the env's 1000.
    _MAX_STEPS = min(
        int(env_params.max_steps_in_episode), args.rows * args.cols + 1
    )

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
            use_prev_action=True,
            critic_layer_norm=True,
            obs_fn=obs_fn, mask_fn=mask_fn,
            burn_in_from_stored_carry=args.burn_in_from_stored_carry,
            use_gvd=args.gvd, gvd_feature_fn=gvd_feature_fn,
            gvd_sf_dims=(128,),
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

    tag = ("SAC+LD" if args.approximate_lambda else "SAC") + (
        "+GVD" if args.gvd else ""
    )
    print(f"Building {tag} agent for Battleship {args.rows}x{args.cols} "
          f"(memory={args.memory_type}, hidden={args.memory_hidden_dim})…")
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
        print(f"Board {args.rows}×{args.cols}, {N} cells.")
        print(f"Collecting {args.collect_steps} steps (auto-reset)…")

        @partial(jax.jit, static_argnames=["n_steps"])
        def collect_rollout(agent_state, key, n_steps):
            key, rk = jax.random.split(key)
            obs, env_st = env.reset(rk, env_params)
            carry = zero_carry()

            def step_fn(scan_carry, _):
                obs, env_st, carry, key = scan_carry
                board = env_st.board.reshape(-1)        # true ships
                hits_misses = env_st.hits_misses.reshape(-1)  # shots so far

                key, sk, ek, eps_key = jax.random.split(key, 4)
                raw, new_carry = fns.predict(
                    agent_state, obs, carry, sk, deterministic=False
                )
                policy_action = jnp.round(raw).astype(jnp.int32)
                # epsilon-greedy over *legal* actions (illegal shots are wasted)
                legal = mask_fn(obs)
                random_action = jax.random.categorical(
                    eps_key, jnp.where(legal > 0, 0.0, -1e9)
                ).astype(jnp.int32)
                use_random = jax.random.uniform(eps_key) < args.collect_epsilon
                action = jnp.where(use_random, random_action, policy_action)
                # predict wrote *its* action into the carry tail; on an
                # epsilon override the executed action differs, so rewrite the
                # tail (cf. RecurrentFeatureExtractor.write_prev_action).
                new_carry = new_carry.at[-N:].set(jax.nn.one_hot(action, N))

                next_obs, next_st, _, done, _ = env.step(
                    ek, env_st, action, env_params
                )
                carry_out = jnp.where(done, zero_carry(), new_carry)
                return (next_obs, next_st, carry_out, key), {
                    # Probe the *memory* part only — the prev-action tail is
                    # an input encoding, not learned state.
                    "carries": new_carry[:CARRY_DIM],
                    "board_masks": board.astype(jnp.float32),
                    "hits_misses": hits_misses.astype(jnp.float32),
                    "dones": done.astype(jnp.float32),
                }

            _, data = jax.lax.scan(step_fn, (obs, env_st, carry, key), length=n_steps)
            return data

        def _collect_and_parse(seed_offset):
            data = collect_rollout(state, jax.random.key(args.seed + seed_offset), args.collect_steps)
            c = np.array(data["carries"])
            b = np.array(data["board_masks"])
            hm = np.array(data["hits_misses"])
            d = np.array(data["dones"])
            di = np.where(d > 0.5)[0]
            es = np.concatenate([[0], di + 1])
            es = es[es < len(c)]
            eb = np.concatenate([es, [len(c)]])
            return c, b, hm, eb

        def _save_dataset(path, c, b, hm, eb):
            print(f"  {len(c)} steps, {len(eb)-1} episodes → {path}")
            with open(path, "wb") as f:
                pickle.dump({"carries": c, "board_masks": b, "hits_misses": hm,
                             "ep_bounds": eb, "rows": args.rows, "cols": args.cols,
                             "tag": tag}, f)

        print(f"Collecting {args.collect_steps} train steps (auto-reset)…")
        carries, board_masks, hits_misses, ep_bounds = _collect_and_parse(1000)
        _save_dataset(dataset_path, carries, board_masks, hits_misses, ep_bounds)

        print(f"Collecting {args.collect_steps} test steps (separate seed)…")
        test_carries, test_board_masks, test_hits_misses, test_ep_bounds = _collect_and_parse(2000)
        _save_dataset(test_dataset_path, test_carries, test_board_masks, test_hits_misses, test_ep_bounds)
    else:
        def _load_dataset(path):
            print(f"Loading dataset ← {path}")
            with open(path, "rb") as f:
                ds = pickle.load(f)
            return (ds["carries"], ds["board_masks"], ds["hits_misses"],
                    ds["ep_bounds"], ds["rows"], ds["cols"], ds.get("tag", tag))

        carries, board_masks, hits_misses, ep_bounds, _rows, _cols, tag = _load_dataset(dataset_path)
        test_carries, test_board_masks, test_hits_misses, test_ep_bounds, _, _, _ = _load_dataset(test_dataset_path)
        print(f"Train: {len(carries)} steps, {len(ep_bounds)-1} eps.  "
              f"Test: {len(test_carries)} steps, {len(test_ep_bounds)-1} eps.")

    # ── Phase 3 — Train probe ────────────────────────────────────────────────

    if not args.skip_probe:
        print(
            f"Training probe ({args.probe_steps} steps, "
            f"hidden={args.probe_hidden_dim}, lr={args.probe_lr})…"
        )
        # Probe has 2N outputs: first N = P(ship), last N = P(cell was fired at).
        # The fired head tests whether the memory retains *which* cells it shot.
        probe_params = init_probe_params(
            jax.random.key(args.seed + 2000),
            CARRY_DIM, 2 * N, args.probe_hidden_dim,
        )
        opt = optax.adam(args.probe_lr)
        opt_state = opt.init(probe_params)

        c_jnp = jnp.array(carries)
        ship_t = jnp.array(board_masks)
        fired_t = (jnp.array(hits_misses) != 0).astype(jnp.float32)
        t_jnp = jnp.concatenate([ship_t, fired_t], axis=-1)  # [ship(N) | fired(N)]

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
            jax.random.key(0), CARRY_DIM, 2 * N, args.probe_hidden_dim
        )
        _, td = jax.tree.flatten(ref)
        probe_params = td.unflatten([jnp.array(l) for l in saved["leaves"]])

    ROWS, COLS = args.rows, args.cols


# ════════════════════════════════════════════════════════════════════════════
#  Vis-only: load dataset + probe (no lambda-imitation needed)
# ════════════════════════════════════════════════════════════════════════════

if args.vis_only:
    print(f"Vis-only mode — loading artefacts from {args.output_dir}")

    with open(dataset_path, "rb") as f:
        ds = pickle.load(f)
    carries = ds["carries"]
    board_masks = ds["board_masks"]
    ROWS, COLS = ds["rows"], ds["cols"]
    tag = ds.get("tag", "SAC+LD")
    N = ROWS * COLS
    CARRY_DIM = carries.shape[1]
    print(f"  train: {len(carries)} timesteps, {ROWS}x{COLS} board, "
          f"{len(ds['ep_bounds'])-1} episodes")

    with open(test_dataset_path, "rb") as f:
        tds = pickle.load(f)
    test_carries = tds["carries"]
    test_board_masks = tds["board_masks"]
    test_hits_misses = tds["hits_misses"]
    test_ep_bounds = tds["ep_bounds"]
    print(f"  test:  {len(test_carries)} timesteps, {len(test_ep_bounds)-1} episodes")

    with open(probe_path, "rb") as f:
        saved = pickle.load(f)
    probe_params = saved["treedef"].unflatten(
        [jnp.array(l) for l in saved["leaves"]]
    )
    print("  probe loaded")


# ════════════════════════════════════════════════════════════════════════════
#  Phase 4 — Visualise (shared rendering code)
# ════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

WATER_COLOR = np.array([0.06, 0.12, 0.28])
SHIP_COLOR = np.array([0.15, 0.85, 0.35])
MISS_COLOR = np.array([0.55, 0.60, 0.70])
HIT_COLOR = np.array([0.95, 0.20, 0.20])
PRED_HIT_COLOR = np.array([1.0, 0.45, 0.10])   # orange: predicted fired + hit
PRED_MISS_COLOR = np.array([0.35, 0.65, 0.95])  # light blue: predicted fired + miss

# Colormaps for the colorbars: water→ship for P(ship), red→green for accuracy.
SHIP_CMAP = LinearSegmentedColormap.from_list("ship", [WATER_COLOR, SHIP_COLOR])
ACC_CMAP = LinearSegmentedColormap.from_list("acc", [(1.0, 0.0, 0.15), (0.0, 1.0, 0.15)])


def _board_legend(fig):
    """Shared legend: cell fill = ship belief, outline = the agent's shots.

    Shots are drawn as OUTLINES (not fills) so the probe's P(ship) underneath a
    fired cell stays visible — otherwise the ground-truth shot colour would hide
    whatever the probe actually predicts there.
    """
    handles = [
        mpatches.Patch(color=WATER_COLOR, label="fill: water (P≈0)"),
        mpatches.Patch(color=SHIP_COLOR, label="fill: ship (P≈1)"),
        mpatches.Patch(facecolor="none", edgecolor=MISS_COLOR, lw=1.6, label="truth outline: miss"),
        mpatches.Patch(facecolor="none", edgecolor=HIT_COLOR, lw=1.6, label="truth outline: hit"),
        mpatches.Patch(facecolor="none", edgecolor=PRED_HIT_COLOR, lw=1.6, ls="--",
                       label="probe outline (dashed): pred. fired, hit"),
        mpatches.Patch(facecolor="none", edgecolor=PRED_MISS_COLOR, lw=1.6, ls="--",
                       label="probe outline (dashed): pred. fired, miss"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))


def to_grid(values):
    """Reshape a length-(ROWS*COLS) vector to the board grid (row-major)."""
    return np.asarray(values).reshape(ROWS, COLS)


def render(ax, ship_grid, fired_grid, title="", pred_fired=False):
    """Fill = ship belief (water→green); fired cells drawn as outlines on top.

    The outline does NOT overwrite the cell colour, so on the probe panel you can
    see whether the probe's belief at a fired cell matches the shot result
    (e.g. green fill under a 'fired' outline = correctly recalled hit).

    ``pred_fired=False`` (truth panel): ``fired_grid`` holds the true hit/miss
    code (1=miss, 2=hit) and outlines are coloured grey/red accordingly (solid).
    ``pred_fired=True`` (probe panel): ``fired_grid`` holds P(fired); any cell
    with P>0.5 gets a DASHED outline — the probe's reconstruction of *which*
    cells were fired at — coloured by the predicted shot result.  A fired cell
    is a hit iff a ship occupies it, so the probe's own P(ship) at that cell IS
    its hit/miss prediction: orange = predicted hit (P(ship)>0.5), light blue =
    predicted miss.
    """
    img = np.empty((ROWS, COLS, 3))
    for r in range(ROWS):
        for c in range(COLS):
            v = float(np.nan_to_num(ship_grid[r, c]))
            img[r, c] = WATER_COLOR * (1 - v) + SHIP_COLOR * v
    ax.imshow(img, interpolation="nearest", aspect="equal")
    for r in range(ROWS):
        for c in range(COLS):
            fv = fired_grid[r, c]
            ls = "solid"
            if pred_fired:
                if fv > 0.5:
                    hit_pred = float(np.nan_to_num(ship_grid[r, c])) > 0.5
                    edge = PRED_HIT_COLOR if hit_pred else PRED_MISS_COLOR
                    ls = (0, (3, 1.5))  # dashed: prediction, not ground truth
                else:
                    continue
            else:
                if fv == 1 or fv == 2:
                    edge = HIT_COLOR if fv == 2 else MISS_COLOR
                else:
                    continue
            # inset slightly so adjacent fired cells don't double their borders
            ax.add_patch(mpatches.Rectangle(
                (c - 0.38, r - 0.38), 0.76, 0.76, fill=False, edgecolor=edge,
                lw=1.3, linestyle=ls,
            ))
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def _vis_stored_episode(ve_idx, probe_params, ds_carries, ds_boards, ds_hm, ds_ep_bounds):
    """Render one stored episode (truth vs probe) as a multi-frame PNG."""
    start, end = ds_ep_bounds[ve_idx], ds_ep_bounds[ve_idx + 1]
    ep_c = ds_carries[start:end]
    ep_b = ds_boards[start:end]
    ep_hm = ds_hm[start:end]
    ep_len = end - start

    preds = np.array(jax.nn.sigmoid(probe_forward(probe_params, jnp.array(ep_c))))
    ship_preds = preds[:, :N]    # P(ship)
    fired_preds = preds[:, N:]   # P(cell was fired at)

    nf = min(args.vis_frames, ep_len)
    idxs = np.linspace(0, ep_len - 1, nf, dtype=int)

    fig, axes = plt.subplots(2, nf, figsize=(2.4 * nf, 6))
    if nf == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(
        f"Board Probe — {tag}  episode {ve_idx + 1}  (len={ep_len})",
        fontsize=10,
    )
    for j, idx in enumerate(idxs):
        render(axes[0, j], to_grid(ep_b[idx]), to_grid(ep_hm[idx]), f"True  t={idx}")
        render(axes[1, j], to_grid(ship_preds[idx]), to_grid(fired_preds[idx]),
               f"Pred  t={idx}", pred_fired=True)
    axes[0, 0].set_ylabel("Ground Truth", fontsize=9)
    axes[1, 0].set_ylabel("Probe", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 0.94, 1])
    # P(ship) colorbar (water→ship green) + discrete shot legend.
    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=SHIP_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("P(ship)  —  truth is 0/1, probe is continuous", fontsize=8)
    _board_legend(fig)
    p = os.path.join(args.output_dir, f"board_probe_ep{ve_idx + 1}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ── episode PNGs (from held-out test episodes) ─────────────────────────────

n_test_eps = len(test_ep_bounds) - 1
n_vis = min(args.vis_episodes, n_test_eps)
print(f"Generating {n_vis} episode visualisation(s) from test set…")
for vi in range(n_vis):
    p = _vis_stored_episode(vi, probe_params, test_carries, test_board_masks,
                            test_hits_misses, test_ep_bounds)
    print(f"  → {p}")

# ── accuracy summary ─────────────────────────────────────────────────────────
#
# A FIRED cell was directly observed by the memory (it chose that shot and saw
# the hit/miss), so decoding its ship/water label tests long-term RETENTION.
# An UNFIRED cell was never observed, so decoding it tests INFERENCE of the
# hidden board.  We report the two separately and, for fired cells, accuracy as
# a function of how long ago the cell was fired (the retention horizon).

print("Computing probe accuracy on held-out test set…")
test_logits = probe_forward(probe_params, jnp.array(test_carries))
test_probs_all = np.array(jax.nn.sigmoid(test_logits))       # (n, 2N)
test_probs = test_probs_all[:, :N]                           # (n, N) P(ship)
test_fired_probs = test_probs_all[:, N:]                     # (n, N) P(fired)
test_preds = (test_probs > 0.5).astype(np.float32)
test_targets = np.asarray(test_board_masks).astype(np.float32)
test_hm = np.asarray(test_hits_misses)
test_fired_targets = (test_hm != 0).astype(np.float32)       # (n, N) 1 = was fired at

correct = (test_preds == test_targets).astype(np.float32)   # (n, N)
fired = test_hm != 0                                        # observed cells
unfired = ~fired                                            # hidden cells


def _auroc(scores, labels):
    """Threshold-free separability: P(score[ship] > score[water]).

    Rank-based (Mann-Whitney U); robust to the 14%/86% class imbalance that
    makes raw accuracy and recall@0.5 misleading.  0.5 = no signal, 1.0 =
    perfectly ranks ships above water.  NaN if a class is absent.
    """
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _acc(mask):
    m = mask.astype(np.float32)
    return float((correct * m).sum() / max(m.sum(), 1.0))


acc = float(correct.mean())
ship_frac = float((test_targets == 1).mean())
majority_baseline = 1.0 - ship_frac  # accuracy of "predict water everywhere"


def _balanced(mask):
    """Balanced accuracy = mean(ship-recall, water-recall) within `mask`.

    Raw accuracy is misleading here: ships are only ~14% of a 10x10 board, so
    'predict water everywhere' already scores ~86%.  Balanced accuracy (and the
    per-class recalls) are the honest signal.
    """
    s = _acc(mask & (test_targets == 1))
    w = _acc(mask & (test_targets == 0))
    return 0.5 * (s + w), s, w


fired_acc = _acc(fired)
unfired_acc = _acc(unfired)
fired_bal, fired_ship, fired_water = _balanced(fired)
unfired_bal, unfired_ship, unfired_water = _balanced(unfired)
frac_fired = float(fired.mean())

# Threshold-free decodability (imbalance-proof): can the probe RANK ships above
# water?  0.5 = no signal regardless of where the 0.5 threshold lands.
fired_auroc = _auroc(test_probs[fired], test_targets[fired])
unfired_auroc = _auroc(test_probs[unfired], test_targets[unfired])

print(f"Ships are {ship_frac:.1%} of the board → majority (all-water) "
      f"baseline = {majority_baseline:.1%} accuracy.")
print(f"Overall: raw={acc:.1%}")
print(f"  Fired   (retention): AUROC={fired_auroc:.3f}  balanced={fired_bal:.1%}  "
      f"raw={fired_acc:.1%}  (hit→ship={fired_ship:.1%}, miss→water={fired_water:.1%})")
print(f"  Unfired (inference): AUROC={unfired_auroc:.3f}  balanced={unfired_bal:.1%}  "
      f"raw={unfired_acc:.1%}  (ship={unfired_ship:.1%}, water={unfired_water:.1%})")
print(f"  (AUROC 0.5 = no decodable signal; raw≈{majority_baseline:.0%} is just the prior.)")
print(f"  Avg {frac_fired:.0%} of cells fired per step.")

# Fired head: can the probe reconstruct *which* cells were fired at?  These were
# directly observed (the agent chose them), so high accuracy here is expected and
# acts as a sanity check that the memory retains its own action history.
test_fired_preds = (test_fired_probs > 0.5).astype(np.float32)
fired_pred_acc = float((test_fired_preds == test_fired_targets).mean())
fired_pred_auroc = _auroc(test_fired_probs, test_fired_targets)
fp_recall = float((test_fired_preds * test_fired_targets).sum()
                  / max(test_fired_targets.sum(), 1.0))      # P(pred fired | fired)
fp_specificity = float(((1 - test_fired_preds) * (1 - test_fired_targets)).sum()
                       / max((1 - test_fired_targets).sum(), 1.0))  # P(pred unfired | unfired)
print(f"Fired prediction (which cells were shot at): AUROC={fired_pred_auroc:.3f}  "
      f"acc={fired_pred_acc:.1%}  (recall={fp_recall:.1%}, specificity={fp_specificity:.1%})")


def _per_cell_acc(mask):
    """Per-cell accuracy averaged over steps where the cell was in `mask`."""
    m = mask.astype(np.float32)
    den = m.sum(0)
    g = np.where(den > 0, (correct * m).sum(0) / np.maximum(den, 1.0), np.nan)
    return to_grid(g)


def _draw_acc(ax, grid, title):
    img = np.empty((ROWS, COLS, 3))
    for r in range(ROWS):
        for c in range(COLS):
            v = grid[r, c]
            img[r, c] = [0.5, 0.5, 0.5] if np.isnan(v) else [1.0 - v, v, 0.15]
    ax.imshow(img, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


fig, (ax_u, ax_f) = plt.subplots(1, 2, figsize=(9, 5.0))
_draw_acc(ax_u, _per_cell_acc(unfired), f"Unfired — inference\n{unfired_acc:.1%}")
_draw_acc(ax_f, _per_cell_acc(fired), f"Fired — retention\n{fired_acc:.1%}")
fig.suptitle(f"Per-Cell Probe Accuracy (test) — {tag}   overall {acc:.1%}", fontsize=11)
sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=ACC_CMAP)
sm.set_array([])
cbar = fig.colorbar(sm, ax=(ax_u, ax_f), fraction=0.046, pad=0.04)
cbar.set_label("per-cell accuracy (grey = never in this category)", fontsize=8)
p = os.path.join(args.output_dir, "board_probe_accuracy.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {p}")

# ── retention horizon: fired-cell accuracy vs steps-since-fired ──────────────
#
# For each fired cell at step t, its "recency" is t − (step it was first fired).
# recency 0 = just fired (trivially observed); larger recency tests how long the
# memory holds the observation.  A smaller memory should decay sooner.

rec_list, prob_list, tgt_list = [], [], []
eb = test_ep_bounds
for i in range(len(eb) - 1):
    s, e = int(eb[i]), int(eb[i + 1])
    if e - s <= 0:
        continue
    hm = test_hm[s:e]            # (L, N)
    prob = test_probs[s:e]      # (L, N) P(ship)
    tgt = test_targets[s:e]     # (L, N) ship(1)/water(0)
    fired_ep = hm != 0
    L = e - s
    any_fired = fired_ep.any(0)
    fire_step = np.where(any_fired, fired_ep.argmax(0), 0)  # first fired step / cell
    rec = np.arange(L)[:, None] - fire_step[None, :]        # (L, N)
    valid = fired_ep & (rec >= 0)
    rec_list.append(rec[valid])
    prob_list.append(prob[valid])
    tgt_list.append(tgt[valid])

if rec_list and sum(len(r) for r in rec_list):
    rec_all = np.concatenate(rec_list)
    prob_all = np.concatenate(prob_list)
    tgt_all = np.concatenate(tgt_list)
    maxr = int(rec_all.max())

    # Threshold-free: mean P(ship) per recency for hit cells (true ship) vs miss
    # cells (true water).  A gap = the memory still encodes the cell; the gap
    # closing as recency grows = forgetting.  Both collapsing to the prior =
    # no decodable signal (independent of any 0.5 threshold).
    def _mean_by_rec(sel):
        r = rec_all[sel]
        p = prob_all[sel]
        s_ = np.bincount(r, weights=p, minlength=maxr + 1)
        n_ = np.bincount(r, minlength=maxr + 1)
        return np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan), n_

    hit_prob, n_hit = _mean_by_rec(tgt_all == 1)
    miss_prob, n_miss = _mean_by_rec(tgt_all == 0)
    x = np.arange(maxr + 1)

    def _xy(y, n, thr=20):
        m = n >= thr
        return x[m], y[m]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(*_xy(hit_prob, n_hit), "-o", ms=3, color=HIT_COLOR,
            label="hit cells (true ship): mean P(ship)")
    ax.plot(*_xy(miss_prob, n_miss), "-o", ms=3, color="#4c72b0",
            label="miss cells (true water): mean P(ship)")
    ax.axhline(ship_frac, ls=":", lw=1, color="grey",
               label=f"prior P(ship)={ship_frac:.2f}")
    ax.set_xlabel("steps since the cell was fired")
    ax.set_ylabel("mean predicted P(ship)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(
        f"Retention horizon — {tag}  (memory={CARRY_DIM}-dim, AUROC={fired_auroc:.2f})\n"
        f"threshold-free: gap between hit/miss curves = decodable memory of the cell",
        fontsize=9,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    p = os.path.join(args.output_dir, "board_probe_retention.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  → {p}")
else:
    print("  (no fired cells collected — skipping retention plot)")

# ── mp4 / gif ────────────────────────────────────────────────────────────────

if args.mp4:
    from matplotlib.animation import FuncAnimation

    if not args.vis_only:
        # Roll out the *deterministic* (greedy, masked) policy for one fresh
        # episode — not the epsilon-soft policy used for data collection.
        print("Rolling out the deterministic policy for one episode (mp4)…")

        @jax.jit
        def _det_rollout(agent_state, key):
            key, rk = jax.random.split(key)
            obs, est = env.reset(rk, env_params)
            carry = zero_carry()

            def step_fn(s, _):
                obs, est, carry, key, alive = s
                key, sk, ek = jax.random.split(key, 3)
                raw, nc = fns.predict(agent_state, obs, carry, sk, deterministic=True)
                a = jnp.round(raw).astype(jnp.int32)
                _nobs, nst, _r, d, _ = env.step(ek, est, a, env_params)
                valid = alive  # 1 up to and including the terminal step, 0 after
                new_alive = alive * (1.0 - d.astype(jnp.float32))
                frame = {
                    # post-step carry (memory part only — the probe's input),
                    # aligned with the post-step board state
                    "carries": nc[:CARRY_DIM],
                    "board": nst.board.reshape(-1).astype(jnp.float32),
                    "hits": nst.hits_misses.reshape(-1).astype(jnp.float32),
                    "valid": valid,
                }
                return (_nobs, nst, nc, key, new_alive), frame

            init = (obs, est, carry, key, jnp.float32(1.0))
            _, data = jax.lax.scan(step_fn, init, length=_MAX_STEPS)
            return data

        data = _det_rollout(state, jax.random.key(args.seed + 7))
        valid = np.array(data["valid"])
        ep_len = int(valid.sum())
        ep_c = np.array(data["carries"])[:ep_len]
        ep_b = np.array(data["board"])[:ep_len]
        ep_hm = np.array(data["hits"])[:ep_len]
        movie_tag = "deterministic policy"
    else:
        # vis-only has no agent/env loaded — fall back to the longest collected
        # (epsilon-soft) test episode.
        print("vis-only: no agent loaded — using the longest collected test episode.")
        test_ep_lens = np.diff(test_ep_bounds)
        best = int(np.argmax(test_ep_lens))
        start, end = int(test_ep_bounds[best]), int(test_ep_bounds[best + 1])
        ep_c = test_carries[start:end]
        ep_b = test_board_masks[start:end]
        ep_hm = test_hits_misses[start:end]
        ep_len = end - start
        movie_tag = "epsilon-soft (collected)"

    ep_preds = np.array(
        jax.nn.sigmoid(probe_forward(probe_params, jnp.array(ep_c)))
    )
    ep_ship_preds = ep_preds[:, :N]    # P(ship)
    ep_fired_preds = ep_preds[:, N:]   # P(fired)

    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(10, 6))
    _sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=SHIP_CMAP)
    _sm.set_array([])
    fig.colorbar(_sm, ax=(ax_t, ax_p), fraction=0.025, pad=0.02).set_label(
        "P(ship)", fontsize=8
    )
    _board_legend(fig)

    def _update(frame):
        ax_t.clear()
        ax_p.clear()
        render(ax_t, to_grid(ep_b[frame]), to_grid(ep_hm[frame]), "Ground Truth")
        render(ax_p, to_grid(ep_ship_preds[frame]), to_grid(ep_fired_preds[frame]),
               "Probe Prediction", pred_fired=True)
        fig.suptitle(f"Board Probe — {tag}  ({movie_tag})   t = {frame}/{ep_len}", fontsize=11)

    print(f"Rendering mp4 ({ep_len} frames, {movie_tag})…")
    anim = FuncAnimation(fig, _update, frames=ep_len, interval=100)

    mp4_path = os.path.join(args.output_dir, "board_probe.mp4")
    try:
        anim.save(mp4_path, writer="ffmpeg", fps=10)
        print(f"  → {mp4_path}")
    except Exception:
        gif_path = os.path.join(args.output_dir, "board_probe.gif")
        anim.save(gif_path, writer="pillow", fps=10)
        print(f"  ffmpeg unavailable, saved gif → {gif_path}")
    plt.close(fig)

print("Done.")
