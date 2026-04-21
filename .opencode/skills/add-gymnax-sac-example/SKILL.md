---
name: add-gymnax-sac-example
description: Add a new pure online SAC example script for a gymnax environment. Covers both discrete and continuous action spaces, placeholder expert buffer, evaluation loop, and optional gymnasium visualisation.
---

## What this skill covers

Adding a standalone SAC example for a **gymnax** environment:

1. `examples/<env_name>_sac.py` — the example script
2. `AGENTS.md` — add entry to Examples section and Commands section

No changes to `src/` or `tests/` are needed.

---

## When to use this skill

- The environment is a **gymnax** environment (uses `gymnax.make()`).
- Training is **pure online SAC** — no expert data, no IQ-Learn.
- You want gymnax for fast JIT-compiled training + optional gymnasium window for post-training visualisation.

---

## Step 1 — extract the env spec

```python
import gymnax
from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax

env, env_params = gymnax.make("<EnvName>")
spec = env_spec_from_gymnax(env, env_params)
```

Key values to look up and comment in the script:
- `spec.obs_shape` — tuple, e.g. `(4,)` for CartPole, `(3,)` for Pendulum
- `spec.action_dim` — int; for discrete this is `num_actions`; for continuous this is `action_dim`
- `spec.is_discrete` — `True` / `False`
- `spec.action_low`, `spec.action_high` — continuous only

---

## Step 2 — placeholder expert buffer

`create_iqlearn_from_env` always requires a structurally valid expert buffer.
When only `fns.train_sac()` is used the buffer is **never sampled**, so the
minimum is 1 all-zeros transition:

```python
import jax.numpy as jnp

expert_data = {
    "observations": jnp.zeros((1, *spec.obs_shape), dtype=jnp.float32),
    "actions":      jnp.zeros((1, 1),              dtype=jnp.float32),
}
```

Set `buffer_size=1` and `batch_size=1` in `Hyperparameters` to match.

---

## Step 3 — Hyperparameters and agent construction

```python
from lambda_imitation.iqlearn import Hyperparameters

hp = Hyperparameters(
    batch_size=1,            # expert batch; never used
    online_batch_size=<B>,   # e.g. 64 or 256
    online_buffer_size=10_000,
    target_entropy=<H>,      # see guidance below
)

state, fns, _ = create_iqlearn_from_env(
    spec,
    expert_data,
    buffer_size=1,
    hp=hp,
    fe_hidden_dims=<FE>,     # e.g. (64, 64) or (128, 128)
    critic_head_dims=<CH>,   # e.g. (64,)
    train_steps=args.train_steps,
)
```

### target_entropy guidance

| Action space | Recommended value | Source |
|---|---|---|
| Discrete (`num_actions` = N) | `0.98 * math.log(N)` | Christodoulou 2019 |
| Continuous (`action_dim` = D) | `-D` (e.g. `-1.0`) | Haarnoja et al. 2018 |

---

## Step 4 — initial reset

```python
key = jax.random.key(args.seed)
key, reset_key = jax.random.split(key)
obs, env_state = env.reset(reset_key, env_params)
```

**gymnax auto-reset**: `env.step()` auto-resets on `done=True` (gymnax base
class always calls `reset_env` internally). Do **not** add a manual
`env.reset` call after `env.step` in the training loop.

---

## Step 5 — evaluation helper

```python
def evaluate(agent_state, rng_key, n_episodes: int = 10) -> float:
    total = 0.0
    for _ in range(n_episodes):
        rng_key, rk = jax.random.split(rng_key)
        ep_obs, ep_state = env.reset(rk, env_params)
        ep_return = 0.0
        done = False
        while not done:
            rng_key, sk = jax.random.split(rng_key)
            action = fns.predict(agent_state, ep_obs, sk, deterministic=True)
            # --- discrete: convert float32 scalar to int32 ---
            # action = jnp.round(action).astype(jnp.int32)
            rng_key, ek = jax.random.split(rng_key)
            ep_obs, ep_state, reward, done, _ = env.step(
                ek, ep_state, action, env_params
            )
            ep_return += float(reward)
            done = bool(done)
        total += ep_return
    return total / n_episodes
```

`fns.predict` return types:
- **Discrete**: `float32` scalar action index — convert with
  `jnp.round(raw).astype(jnp.int32)` before passing to gymnax.
- **Continuous**: `float32` array of shape `(action_dim,)` — pass directly.

---

## Step 6 — training loop

```python
for rnd in range(1, args.rounds + 1):
    key, train_key = jax.random.split(key)
    state, env_state, metrics = fns.train_sac(
        state, env, env_params, env_state, train_key
    )

    key, eval_key = jax.random.split(key)
    mean_return = evaluate(state, eval_key)
    print(
        f"Round {rnd:4d}/{args.rounds}  "
        f"mean_return={mean_return:8.1f}  "
        f"alpha={float(metrics['alpha']):.4f}  "
        f"critic_loss={float(metrics['critic_loss']):.4f}"
    )
```

`fns.train_sac` metric keys: `"q"`, `"entropy"`, `"v"`, `"critic_loss"`,
`"target_q"`, `"alpha"`. The key is `"critic_loss"` (not `"sac_critic_loss"`).

Note the first round triggers JIT compilation — print a warning to the user:
```python
print("(First round JIT-compiles the update loop — expect a short delay.)\n")
```

---

## Step 7 — gymnasium visualisation (optional)

After training, open a gymnasium render window. Wrap in a `try/except
ImportError` so the script still works without gymnasium:

```python
try:
    import gymnasium as gym
except ImportError:
    print(
        "\ngymnasium not installed — skipping visualisation.\n"
        "Install with:  pip install 'gymnasium[classic-control]'"
    )
    sys.exit(0)

vis_env = gym.make("<gymnasium-env-id>", render_mode="human")
obs, _ = vis_env.reset()
key = jax.random.key(args.seed + 1)

while True:
    key, subkey = jax.random.split(key)
    action = fns.predict(state, jnp.array(obs), subkey, deterministic=True)
    # discrete:   vis_env.step(int(jnp.round(action)))
    # continuous: vis_env.step(np.array(action))   ← import numpy as np
    obs, _, terminated, truncated, _ = vis_env.step(<converted action>)
    if terminated or truncated:
        obs, _ = vis_env.reset()
        input("Episode done — press Enter to continue…")
```

---

## Step 8 — CLI (standard argparse block)

All example scripts expose the same three flags:

```python
import argparse

parser = argparse.ArgumentParser(
    prog="<script_name>.py",
    description="<one-line description>",
)
parser.add_argument("--rounds",      type=int, default=<N>,   metavar="N", help="...")
parser.add_argument("--train-steps", type=int, default=<S>,   metavar="N", help="...")
parser.add_argument("--seed",        type=int, default=0,     metavar="S", help="...")
args = parser.parse_args()
```

Parse `args` **before** importing JAX/gymnax so `--help` exits immediately.

---

## Step 9 — update AGENTS.md

Add entries to **two** sections:

### Commands section
```
- Train + visualise <EnvName> SAC (gymnax): `python examples/<env_name>_sac.py`
```

### Examples section
> **`<env_name>_sac.py`**: Pure online SAC (no expert data) on gymnax
> <EnvName>. Require `gymnax` for training; `gymnasium[classic-control]` is
> optional and only needed for the post-training visualisation window. CLI
> flags: `--rounds`, `--train-steps`, `--seed`.

---

## Checklist

- [ ] `examples/<env_name>_sac.py` created
- [ ] `argparse` block at top; `args` parsed before JAX imports
- [ ] `gymnax` import guarded with `try/except ImportError` + `sys.exit`
- [ ] `env_spec_from_gymnax` used; obs_shape / action_dim commented
- [ ] Placeholder expert buffer: 1 zeros transition, `buffer_size=1`, `batch_size=1`
- [ ] `target_entropy` set appropriately (discrete: `0.98*log(N)`; continuous: `-D`)
- [ ] JIT warning printed before training loop
- [ ] `metrics['critic_loss']` used (not `'sac_critic_loss'`)
- [ ] Discrete actions converted via `jnp.round(...).astype(jnp.int32)` for gymnax
- [ ] Continuous actions passed as `float32 (action_dim,)` array
- [ ] Gymnasium visualisation block guarded with `try/except ImportError`
- [ ] `AGENTS.md` Commands and Examples sections updated
- [ ] Script runs without errors (`python examples/<env_name>_sac.py --rounds 1`)
