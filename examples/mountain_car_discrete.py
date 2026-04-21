"""IQ-Learn demo: MountainCar-v0 (discrete action space).

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
IQ-Learn imitation-learning demo on MountainCar-v0 (discrete actions).

Workflow
--------
  Step 1 — record expert demonstrations:

      python mountain_car_discrete.py --record [--episodes N] [--file PATH]

      Opens a pygame window.  Drive the car with the keyboard for N episodes;
      the (observation, action) pairs are saved to a .npz file.

      Controls
        a / left arrow  push left   (action 0)
        s               no push     (action 1) [held default]
        d / right arrow push right  (action 2)

  Step 2 — train and visualise:

      python mountain_car_discrete.py [--steps N] [--file PATH]

      Loads the saved demonstrations, builds an IQ-Learn agent using
      create_iqlearn_from_env, trains for N gradient steps (the first call
      JIT-compiles the update loop), then opens a render window.
      Press Enter in the terminal after each episode; Ctrl-C to quit.
"""

parser = argparse.ArgumentParser(
    prog="mountain_car_discrete.py",
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
    default="mountain_car_discrete_demos.npz",
    metavar="PATH",
    help="path to the demo .npz file (default: mountain_car_discrete_demos.npz)",
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
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    env = RecorderWrapper(env, buffer_size=10_000)

    config = {
        "max_fps": 60,
        "default_action": 1,  # no push when no key held
        "wait_for_input": False,
        "display_size": (800, 600),
        "reset_input": ["a", "s", "d"],
        "keymap": {
            "a": 0,  # push left
            "s": 1,  # no push
            "d": 2,  # push right
            "left": 0,
            "right": 2,
        },
        "joystick": {"axis": [], "keys": []},
    }

    print(f"Recording {args.episodes} episode(s).")
    print("Controls:  a / ← = push left   s = no push   d / → = push right")
    ManualControl(env, config).run(args.episodes)

    n = min(env.pos, env.buffer_size)
    np.savez(
        args.file,
        observations=env.observations[:n].astype(np.float32),
        # Discrete actions are stored as int32 scalars; IQ-Learn expects
        # float32 action indices of shape (N, 1).
        actions=env.actions[:n].reshape(-1, 1).astype(np.float32),
    )
    print(f"Saved {n} transitions to '{args.file}'.")
    print(f"Next step:  python {parser.prog} --steps {args.steps} --file {args.file}")

# ── train + visualise mode ────────────────────────────────────────────────────

else:
    print("IQ-Learn demo — MountainCar-v0 (discrete)")
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
    ref_env = gym.make("MountainCar-v0")
    spec = env_spec_from_gymnasium(ref_env)
    ref_env.close()

    print("Building IQ-Learn agent…")
    state, fns, _ = create_iqlearn_from_env(
        spec,
        expert_data,
        train_steps=args.steps,
    )

    # --- train ---
    print(f"Training for {args.steps} gradient steps (first call JIT-compiles)…")
    state, metrics = fns.train(state, jax.random.key(0))
    print("Training complete.")
    print("Metrics:", {k: f"{float(v):.4f}" for k, v in metrics.items()})

    # --- visualise ---
    print(
        "\nVisualising.  Press Enter in this terminal after each episode to continue."
    )
    print("Ctrl-C to quit.\n")

    env = gym.make("MountainCar-v0", render_mode="human")
    obs, _ = env.reset()
    key = jax.random.key(1)

    while True:
        key, subkey = jax.random.split(key)
        action = fns.predict(state, jnp.array(obs), subkey, deterministic=False)
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            obs, _ = env.reset()
            input("Episode done — press Enter to continue…")
