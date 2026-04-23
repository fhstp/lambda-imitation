"""IQ-Learn demo: MountainCarContinuous-v0 (continuous action space).

Run without arguments to train and visualise, or pass --record to capture
fresh expert demonstrations first.  See --help for the full workflow.

Requirements
------------
    pip install "lambda-imitation[gymnasium]"
    pip install imitation-gym-wrappers   # for ManualControl / RecorderWrapper
"""

import argparse
import sys

# ── CLI ───────────────────────────────────────────────────────────────────────

_DESCRIPTION = """\
IQ-Learn imitation-learning demo on MountainCarContinuous-v0 (continuous actions).

Workflow
--------
  Step 1 — record expert demonstrations:

      python mountain_car_continuous.py --record [--episodes N] [--file PATH]

      Opens a pygame window.  Drive the car with the keyboard for N episodes;
      the (observation, action) pairs are saved to a .npz file.

      Controls
        a   full push left  (action -1.0) [default is 0.0 when no key held]
        d   full push right (action +1.0)

      Tip: build momentum by swinging back and forth before reaching the flag.

  Step 2 — train and visualise:

      python mountain_car_continuous.py [--steps N] [--file PATH]

      Loads the saved demonstrations, builds an IQ-Learn agent using
      create_iqlearn_from_env (action_scale and action_bias are set
      automatically from the environment's action bounds), trains for N
      gradient steps (the first call JIT-compiles the update loop), then
      opens a render window.  Press Enter in the terminal after each episode;
      Ctrl-C to quit.
"""

parser = argparse.ArgumentParser(
    prog="mountain_car_continuous.py",
    description=_DESCRIPTION,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--record",
    action="store_true",
    help="record expert demonstrations instead of training (step 1)",
)
parser.add_argument(
    "--episodes",
    type=int,
    default=5,
    metavar="N",
    help="number of episodes to record (default: 5, record mode only)",
)
parser.add_argument(
    "--steps",
    type=int,
    default=5000,
    metavar="N",
    help="gradient steps per fns.train() call (default: 1000, train mode only)",
)
parser.add_argument(
    "--file",
    default="mountain_car_continuous_demos.npz",
    metavar="PATH",
    help="path to the demo .npz file (default: mountain_car_continuous_demos.npz)",
)
parser.add_argument(
    "--wandb",
    action="store_true",
    help="enable Weights & Biases logging (train mode only)",
)
parser.add_argument(
    "--wandb-project",
    default="lambda-imitation",
    metavar="PROJECT",
    help="W&B project name (default: lambda-imitation)",
)
parser.add_argument(
    "--wandb-run-name",
    default=None,
    metavar="NAME",
    help="W&B run name (default: auto-generated)",
)
args = parser.parse_args()

# ── shared imports ────────────────────────────────────────────────────────────

import jax
import jax.numpy as jnp
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    sys.exit("gymnasium is required.  Install with:  pip install gymnasium")

try:
    from imitation_gym_wrappers.manual_control import ManualControl
    from imitation_gym_wrappers.recorder_wrapper import RecorderWrapper
except ImportError:
    sys.exit(
        "imitation-gym-wrappers is required.\n"
        "Install from: https://git.nwt.fhstp.ac.at/lbkietreiber/imitation-gym-wrappers"
    )

from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnasium

# ── recording mode ────────────────────────────────────────────────────────────

if args.record:
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    env = RecorderWrapper(env, buffer_size=10_000)

    config = {
        "max_fps": 60,
        "default_action": [0.0],  # no force when no key held
        "wait_for_input": False,
        "display_size": (800, 600),
        "reset_input": ["a", "d"],
        "keymap": {
            "a": [{"dim": 0, "value": -1.0}],  # push left
            "d": [{"dim": 0, "value": 1.0}],  # push right
        },
        "joystick": {"axis": [], "keys": []},
    }

    print(f"Recording {args.episodes} episode(s).")
    print("Controls:  a = push left (-1.0)   d = push right (+1.0)   [no key = 0.0]")
    ManualControl(env, config).run(args.episodes)

    n = min(env.pos, env.buffer_size)
    np.savez(
        args.file,
        observations=env.observations[:n].astype(np.float32),
        # RecorderWrapper stores Box actions as float32 shape (buf_size, 1); use as-is.
        actions=env.actions[:n].astype(np.float32),
    )
    print(f"Saved {n} transitions to '{args.file}'.")
    print(f"Next step:  python {parser.prog} --steps {args.steps} --file {args.file}")

# ── train + visualise mode ────────────────────────────────────────────────────

else:
    print("IQ-Learn demo — MountainCarContinuous-v0 (continuous)")
    print("  Run with --help to see the full workflow.")
    print()

    # --- load expert data ---
    try:
        data = np.load(args.file)
    except FileNotFoundError:
        sys.exit(
            f"Demo file '{args.file}' not found.\n"
            f"Record expert demonstrations first:  python {parser.prog} --record"
        )

    expert_data = {k: jnp.array(v) for k, v in data.items()}
    n_demos = expert_data["observations"].shape[0]
    print(f"Loaded {n_demos} expert transitions from '{args.file}'.")

    # --- build agent via the high-level factory ---
    ref_env = gym.make("MountainCarContinuous-v0")
    spec = env_spec_from_gymnasium(ref_env)
    ref_env.close()

    print("Building IQ-Learn agent…")
    state, fns, _ = create_iqlearn_from_env(
        spec,
        expert_data,
        train_steps=args.steps,
    )

    # --- wandb (optional) ---
    _wandb = None
    if args.wandb:
        try:
            import wandb as _wandb_mod
            _wandb = _wandb_mod
        except ImportError:
            print("wandb not installed — skipping.  Install with:  pip install wandb")
        else:
            _wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "env": "MountainCarContinuous-v0",
                    "algo": "IQ-Learn",
                    "action_space": "continuous",
                    "train_steps": args.steps,
                    "n_expert_transitions": n_demos,
                },
            )

    # --- train ---
    print(f"Training for {args.steps} gradient steps (first call JIT-compiles)…")
    state, metrics = fns.train(state, jax.random.key(0))
    print("Training complete.")
    print("Metrics:", {k: f"{float(v):.4f}" for k, v in metrics.items()})

    if _wandb is not None:
        _wandb.log({k: float(v) for k, v in metrics.items()})
        _wandb.finish()

    # --- visualise ---
    print(
        "\nVisualising.  Press Enter in this terminal after each episode to continue."
    )
    print("Ctrl-C to quit.\n")

    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    obs, _ = env.reset()
    key = jax.random.key(1)

    while True:
        key, subkey = jax.random.split(key)
        action = fns.predict(state, jnp.array(obs), subkey, deterministic=False)
        # action has shape (1,); env.step expects a numpy array of the same shape.
        obs, _, terminated, truncated, _ = env.step(np.array(action))
        if terminated or truncated:
            obs, _ = env.reset()
            input("Episode done — press Enter to continue…")
