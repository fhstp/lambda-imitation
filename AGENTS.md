# AGENTS.md

## Overview

IQ-Learn (Inverse Q-Learning) imitation learning implementation in **Python** using **JAX + Flax NNX + Optax**. Currently on the `fresh_start` branch. The project is structured as an installable Python package (`lambda-imitation`) with a `src/` layout.

## Repository layout

```
src/
    lambda_imitation/
        __init__.py       ← public API re-exports
        buffer.py         ← generic circular replay buffer
        iqlearn.py        ← SAC-style actor-critic with IQ-Learn reward recovery
        utils.py          ← environment-interface adapters and high-level factory
tests/
    __init__.py
    test_buffer.py        ← 26 tests
    test_iqlearn.py       ← 90 tests
    test_utils.py         ← 45 tests
examples/
    mountain_car_discrete.py     ← MountainCar-v0 demo (discrete, gymnasium, IQ-Learn)
    mountain_car_continuous.py   ← MountainCarContinuous-v0 demo (continuous, gymnasium, IQ-Learn)
    cartpole_sac.py              ← CartPole-v1 demo (discrete, gymnax, pure SAC)
    pendulum_sac.py              ← Pendulum-v1 demo (continuous, gymnax, pure SAC)
pyproject.toml            ← hatchling build, optional extras, pytest config
```

## Architecture

- `buffer.py` -- Generic circular replay buffer using JAX arrays and NamedTuples
- `iqlearn.py` -- SAC-style actor-critic with IQ-Learn reward recovery (configurable networks, continuous and discrete actions)
- `utils.py` -- Environment-interface adapters: extracts `EnvSpec` from gymnasium / gymnax / jumanji environments, plus `create_iqlearn_from_env` high-level factory

**Design pattern:** Purely functional. Both modules use a factory pattern (`create_buffer`, `create_iqlearn`) that returns `(state_namedtuple, functions_namedtuple)`. No mutable class state. Flax NNX models are split into graph definition + state via `nnx.split`/`nnx.merge` for functional updates inside `jax.jit`.

### Network design (iqlearn.py)

Networks are split into a **feature extractor** and a **head**:

- `MLPFeatureExtractor(input_dim, hidden_dims, rngs)` -- configurable MLP backbone; flattens input, applies ReLU layers. `hidden_dims=(256, 256)` by default.
- `Head(feature_dim, hidden_dims, output_dim, *, rngs)` -- generic head: ReLU on hidden layers, linear output. No flattening. Used for all four roles:
  - continuous actor: `output_dim = 2 * action_dim` (mean + log_std)
  - discrete actor: `output_dim = num_actions` (categorical logits)
  - continuous critic Q1/Q2: `output_dim = 1` (actions concatenated to features before the head)
  - discrete critic Q1/Q2: `output_dim = num_actions` (per-action Q-values)

FE + head state are stored together in a `NetworkState(fe, head)` NamedTuple. The twin-Q critic uses `TwinCriticState(q1: NetworkState, q2: NetworkState)` — both branches are fully independent (separate FE and head) and a single optimizer operates on the whole pytree.

`create_iqlearn` takes three separate FE instances (`actor_feature_extractor`, `critic_q1_feature_extractor`, `critic_q2_feature_extractor`). It infers each feature dim by running a dummy forward pass before splitting. Use plain Python `list` (not `nnx.List`) for collections of sub-modules inside an `nnx.Module` — `nnx.List` was removed in Flax 0.10.7.

Graph definitions are stored in `NetworkGraphs(fe, head)` NamedTuples, grouped into `IQLearnGraphs(actor, critic_q1, critic_q2)`.

### Action space support

Both **continuous** (Box-style) and **discrete** (categorical) action spaces are supported. Pass `is_discrete=True` to `create_iqlearn` / `create_iqlearn_from_env` to activate the discrete path:

- Discrete actor: categorical policy; `predict` returns a `float32` scalar action index (e.g. `0.0`, `1.0`).
- Discrete critic: all-actions Q-values; V(s) computed as exact `Σ_a π(a|s)·Q(s,a)` without sampling.
- Default `target_entropy` for discrete: `0.98 * log(num_actions)` (Christodoulou 2019).
- `MultiDiscreteArray` (jumanji) is not supported — raises `ValueError`.

## Dependencies

Declared in `pyproject.toml`. Install with:

```
pip install -e ".[dev]"          # core + pytest
pip install -e ".[dev,gymnasium]"  # add gymnasium support
```

Core: `jax>=0.4.30`, `jaxlib>=0.4.30`, `flax>=0.9.0`, `optax>=0.2.0`, `numpy>=1.26`  
Optional extras: `[gymnasium]`, `[gymnax]`, `[jumanji]`, `[dev]` (pytest)  
Requires Python 3.10+ (uses `X | Y` union type syntax).

## Commands

- Install: `pip install -e ".[dev]"`
- Run all tests: `pytest tests/`
- Run buffer tests only: `pytest tests/test_buffer.py`
- Run iqlearn tests only: `pytest tests/test_iqlearn.py`
- Run utils tests only: `pytest tests/test_utils.py`
- Record demos (discrete): `python examples/mountain_car_discrete.py --record`
- Train + visualise (discrete): `python examples/mountain_car_discrete.py`
- Record demos (continuous): `python examples/mountain_car_continuous.py --record`
- Train + visualise (continuous): `python examples/mountain_car_continuous.py`
- Train + visualise CartPole SAC (gymnax): `python examples/cartpole_sac.py`
- Train + visualise Pendulum SAC (gymnax): `python examples/pendulum_sac.py`
- Train PocMan SAC+λ (lambda-envs): `python examples/lambda-envs/pocman_sac_mc.py`
- Partially-observable CartPole SAC+λ: `python examples/lambda-envs/cartpole_partial_sac.py --obs-noise 0.1`
- RockSample SAC+λ with a rock-goodness probe: `python examples/lambda-envs/rocksample_probe.py --config rocksample_11_11`
- Train Battleship SAC+λ, action-masked (lambda-envs): `python examples/lambda-envs/battleship_sac_mc.py`
- Pellet probe + visualisation (PocMan): `python examples/lambda-envs/pocman_pellet_probe.py`
- Multi-seed PocMan probe with an expert warm start: `python examples/lambda-envs/pocman_pellet_probe.py --num-seeds 3 --expert-prefill-steps 20000`
- Offline PocMan training + probe (no env interaction): `python examples/lambda-envs/pocman_pellet_probe.py --offline --expert-prefill-steps 100000 --num-seeds 3`
- Scripted PocMan expert, evaluated against random: `python examples/lambda-envs/pocman_expert.py --episodes 256`
- T-maze behaviour/target Retrace-discrepancy comparison (exact): `python examples/lambda-envs/tmaze_lambda_discrepancy.py`
- Figure-3-style T-maze observability sweeps: `python examples/lambda-envs/tmaze_retrace_aliasing.py`
- Local on-policy λ-discrepancy turnaround sweep: `python examples/lambda-envs/tmaze_lambda_turnaround.py`
- Board probe + visualisation (Battleship): `python examples/lambda-envs/battleship_board_probe.py`
- Offline Bayes-data training + probe (Battleship): `python examples/lambda-envs/battleship_board_probe.py --offline --expert-prefill-steps 100000 --num-seeds 3`
- Online with an expert warm start: `python examples/lambda-envs/battleship_board_probe.py --expert-prefill-steps 20000 --num-seeds 3`

## Conventions

- All state is immutable NamedTuples; updates return new instances.
- Buffer entries are keyed by string names in `info: dict[str, jax.Array]`, not fixed fields.
- Hyperparameters are a NamedTuple with defaults, not a dataclass.
- `jax.lax.scan` is used for multi-step training loops; avoid Python for-loops inside JIT-compiled code.
- Actor/Critic FEs flatten input via `x.reshape(x.shape[0], -1)` to handle any obs shape.
- `utils.py` extractors do lazy imports (each library only imported when its extractor is called); only the library you use needs to be installed.
- `utils.py` supports both continuous (Box-style) and discrete action spaces; discrete actions in expert data are stored as float32 indices of shape `(N, 1)`.
- Mocking gymnax/jumanji in tests: use `ModuleType` objects (not `MagicMock`) for the full parent-module chain and wire `.spaces`/`.specs` attributes explicitly, so CPython's attribute-traversal in dotted imports resolves to the mock classes rather than auto-generated MagicMock attributes.
- All imports within the package use relative imports (`from .buffer import ...`).
- The twin-Q `jnp.min` gradient flow: with continuous actions, only the branch producing the smaller Q receives gradient in a given step. Tests for per-branch parameter changes therefore check the full `TwinCriticState` pytree (not individual branches), and verify branch independence by checking initial parameter divergence rather than per-step updates.
- **Gymnax auto-reset**: the gymnax base class `Environment.step()` always auto-resets — it runs `reset_env` on every call and returns the reset state/obs when `done=True` (via `jax.lax.select`). Do NOT add a manual `env.reset` call after `env.step` in gymnax code; `run_env_step` in `iqlearn.py` relies on this. Mock gymnax environments used in SAC tests must replicate this behaviour in their `step()` method.
- **Simulating absent installed packages in tests**: deleting a package from `sys.modules` is not sufficient when the package is actually installed — Python will re-import it from disk. Use `patch.dict(sys.modules, {key: None, ...})` (setting entries to `None`) to block re-import and trigger `ImportError`, as `None` entries are treated as import blockers by the import machinery.
- **`fns.train_sac()` metric keys**: `"q"`, `"entropy"`, `"v"`, `"critic_loss"`, `"target_q"`, `"alpha"`. Note the SAC critic loss key is `"critic_loss"` (same as in `update_step`), not `"sac_critic_loss"`.
- **Pure SAC examples with `create_iqlearn_from_env`**: when only `fns.train_sac()` is used (no imitation learning), a placeholder expert buffer is required. Use 1 all-zeros transition with `buffer_size=1` and `batch_size=1` in `Hyperparameters` — this is structurally valid and never sampled.
- **Prev-action input (`use_prev_action`)**: the FE call convention is `feature_extractor(carry, obs, prev_action=None) -> (new_carry, y)`. The carry is the **memory state only** — the prev-action is *not* packed into it. When `create_iqlearn(_from_env)` is built with `use_prev_action=True` the previously executed action (one-hot for discrete, squashed action values for continuous) is fed to the **projection** as an explicit input and threaded *next to* the carry: `run_env_step`/`train_unrolled` thread an `env_prev_action` (internally, so their public signatures are unchanged), `calculate_latent` threads `enc(a_{t-1})` in its local scan carry, and both reset to zero at episode boundaries. The default `LinearProjection` reproduces the old behaviour exactly (concat `[flatten(obs), prev_action]` then one `Linear` — matching `ActionConcatWrapper` in the original lambda-discrepancy code, action through the embedding never the cell). `carry_dim` is memory-only — hand-rolled `CARRY_DIM` formulas in examples must **not** add `action_dim`; thread a separate `zero_prev_action()` instead. `predict` gains an optional keyword `prev_action=` and returns the memory carry only; manual rollout loops compute the next prev-action from the *executed* action via `fns.encode_action(...)` (this transparently handles epsilon-greedy overrides — encode whatever was executed). The **projection is configurable** (`projection=` on `create_iqlearn_from_env`, `RecurrentFeatureExtractor`): an `int` (default `LinearProjection`), `None` (no-embedding concat), or a builder `(obs_shape, prev_action_dim, rngs) -> nnx.Module` with contract `module(obs, prev_action) -> z` receiving raw (unflattened) obs — see `BattleshipProjection` in `lambda-envs/battleship_sac_mc.py` (`--skip-projection`) for a spatial skip-connection example. With `burn_in_from_stored_carry` the prev-action input is stored in the buffer under `prev_actions` (alongside `carries`) and seeds the burn-in's first step. Tests: `tests/test_prev_action.py` (includes the obs-concat equivalence test and a custom-projection regression).
- **Offline training hooks**: `fns.prefill_buffer(..., behaviour_fn=fn)` fills the buffer from a scripted policy instead of the uniform-random default — `fn(obs, env_state, key) -> (action_index, b(a|s))`, discrete only, static (same callable object across calls or it recompiles); the returned probability is stored under `behaviour_key` and divides the V-trace ratios. `fns.update_only(state, n_steps, key) -> (state, metrics)` runs gradient updates off the buffer with no env at all (the offline counterpart of `fns.train`; same `update_step`, same metric keys). Tests: `tests/test_offline_prefill.py`.
- **Action masking (`obs_fn` / `mask_fn`)**: discrete only. The buffer always stores the *full* observation; `obs_fn` (default identity) selects what the FE sees, and `mask_fn` (default `None`) derives a legal-action mask from the same full observation. The mask is applied wherever logits become a categorical distribution: `predict`, `get_v`, `get_entropy`, `get_importance_ratios`, and the random prefill policy (which then samples uniformly over legal actions with `behaviour_prob = 1/num_legal`). Masked logits are filled with a large *finite* negative (`-1e9`), **not** `-inf`: `-inf` makes the `probs·log_probs` entropy term back-propagate `NaN` (the `jnp.where`-with-`inf` trap) into the shared FE; `-1e9` keeps masked probabilities at ~0 with finite gradients. Fully-masked rows (zero-padded burn-in obs) are left unmasked to keep softmax well defined.

## Examples

Scripts in `examples/` are standalone demos — not collected by pytest.

- **`lambda-envs/tmaze_retrace_aliasing.py`**: Figure-3-style extension of the exact T-maze Retrace experiment. Holds each behaviour/target pair fixed while interpolating `Phi_eta = (1-eta) I + eta Phi_aliased` for corridor, junction, or both. Six default pairs; `--pairs behaviour:target_p` selects arbitrary combinations (fractions accepted). Generates individual original-layout figures and shared-axis overviews for the main hallway and one-cell control, plus SVGs and JSON. Intermediate observations are stochastic: augment latent state to `(s,o)` before calling the deterministic-observation evaluator, never simply replace its `phi` by a stochastic matrix. Fixed latent-pair scoring weights are transported through the observation channel; the both-aliased endpoint matches the five-pair RMS from `tmaze_retrace_behaviours`. Defaults retain γ=0.9 and RMS rather than the original Figure 3's γ≈1/max norm. Reversed and uniform-random behaviours can have a nonzero interior signal but a zero fully aliased endpoint. Details: `tmaze_retrace_discrepancy.md`; independent stochastic-history enumeration and endpoint checks: `tests/test_tmaze_retrace_aliasing.py`.
- **`lambda-envs/tmaze_lambda_turnaround.py`**: exact local on-policy λ-discrepancy counterexample for the Figure-3 observation interpolation. A hand-constructed one-cell policy has `TD(0)=TD(1)` at both perfect and fully both-aliased endpoints but a nonzero interior discrepancy. Sweeps nearby cue, hallway and junction probabilities: they retain nonzero fully-aliased discrepancies but all turn around near η≈0.13–0.21; their squared-loss derivative is negative over most of the path, meaning gradient descent in the scalar aliasing parameter would discard information. This is a diagnostic for global objective alignment, not a claim that an RNN parameter gradient is the same as `d/dη`. Outputs: `tmaze_ld_output/tmaze_lambda_turnaround_*`; checks: `tests/test_tmaze_lambda_turnaround.py`.

- **`mountain_car_discrete.py` / `mountain_car_continuous.py`**: IQ-Learn imitation learning on gymnasium MountainCar environments. Require `gymnasium` and `imitation-gym-wrappers`. Two-step workflow: `--record` to capture expert demos, then train and visualise.
- **`cartpole_sac.py` / `pendulum_sac.py`**: Pure online SAC (no expert data) on gymnax CartPole-v1 and Pendulum-v1. Require `gymnax` for training; `gymnasium[classic-control]` is optional and only needed for the post-training visualisation window. CLI flags: `--rounds`, `--train-steps`, `--seed`.
- **`lambda-envs/cartpole_partial_sac.py`**: gymnax CartPole-v1 with the two velocity components masked out (obs = `[x, theta]`), on the shared runner. `--obs-noise` adds Gaussian noise as a *fraction of each dimension's range*, which is what makes the memory non-trivial: clean, the velocity is recoverable from two consecutive observations, so plain SAC solves it; noisy, it has to be averaged over ~4–10 steps. `--full-obs` is the MDP control, `--memory-type identity` the memoryless floor. Its probe is the only **regression** one — it decodes the full 4-dim state from the carry under MSE and reports per-dimension R2 (`r2_x_dot`/`r2_theta_dot` are the masked velocities and the actual measurement; `r2_x`/`r2_theta` are observed and only check the wiring). Targets are z-scored before the MSE, or theta's smaller scale would be ignored. Defaults are small on purpose (32-unit GRU, 64-wide heads, batch 64, sequence 5 / burn-in 3, a 1000-step round): 200 rounds × 5 seeds takes ~17 min on one GPU.
- **`lambda-envs/rocksample_probe.py`**: RockSample (`--config {5_5,7_8,11_11,15_15}`, γ and hidden size taken from the config) with a two-head probe for rock goodness and whether each rock was sampled, split by `checked`. `--use-prev-action` defaults to **False** here, matching the reference: each rock has its own observation slot, so the action stream is not needed to disambiguate. α is the sensitive knob — at 0.1 the policy entropy collapses and the critic diverges; 0.5 works at 5×5. Walking east exits with +10, so 10 is the do-nothing floor.
- **`lambda-envs/pocman_sac_mc.py` / `lambda-envs/battleship_sac_mc.py`**: Pure online SAC + λ-discrepancy on `lambda-envs` POMDPs (require `lambda-envs`). Multi-seed, vmapped/jitted training with a recurrent FE (`--memory-type`). **Battleship needs action masking**: the env packs a legal-action mask into the observation tail, and the demo splits it via `obs_fn`/`mask_fn` passed to `create_iqlearn_from_env` — `obs_fn=obs[...,:1]` (only the last-shot bit reaches the FE), `mask_fn=obs[...,1:]` (the mask applied to the policy). `mask_fn` defaults to `None` (no masking) everywhere else.
- **`lambda-envs/battleship_board_probe.py` offline / expert-prefill modes**: `--expert-prefill-steps N` fills the replay buffer from the scripted **Bayes-density player** (count every ship placement consistent with the observed misses, weight `--hit-weight` per covered hit, fire the densest unfired cell; `--expert-prefill-epsilon`, default 0.1) instead of the uniform-random prefill. `--offline` then runs `--rounds × --train-steps` **gradient updates with no environment interaction** on that buffer — the multi-seed vmapping, probe-evals, W&B bands and checkpointing are the same code as the online path. `--probe-rollout-policy bayes` probes along the scripted player's trajectories rather than the actor's (the distribution an offline agent actually trained on). `--save-fe-every-eval` / `--init-fe` save and transplant a trained memory, which with `--fe-lr 0` freezes it and trains only the heads. Measured on 5×5 (3,2): a 20k expert prefill plus online training reaches Bayes-level play (~13 shots) at ~70k env steps and holds it, against ~1M for the reference PPO+LD.
- **`lambda-envs/tmaze_lambda_discrepancy.py`**: exact **Retrace-discrepancy** comparison with independently varied behaviour and target policies. Five fixed behaviours (original two-thirds-up, balanced, reversed, paper + 50% random, uniform random) evaluate forward-moving targets swept over junction probabilities. Solves the paper's TD-residual Retrace fixed point with a fixed discounted-occupancy posterior per behaviour; unsupported coordinates are removed, never ridge-regularised. Defaults: five-cell reference maze, γ=0.9, λ=(0,1), plus a **one-cell control** where balanced and uniform-random behaviours have exactly zero own λ-discrepancy without changing γ. Balanced behaviour nevertheless detects biased targets; reversed behaviour has a nonzero own discrepancy but misses the two-thirds-up target; uniform random misses the entire target interval [1/4,3/4] in the control. The plotted norm is RMS over five fixed shared observation-action pairs, not the old policy-weighted squared gap. On-policy values are a reference, not the "true" off-policy discrepancy. PNG/SVG figures and numerical JSON go to `tmaze_ld_output/tmaze_retrace_*`. NumPy + optional Matplotlib only. Details: `examples/lambda-envs/tmaze_retrace_discrepancy.md`; independent trajectory, Markov-control, Monte Carlo endpoint and real-env transition checks: `tests/test_tmaze_retrace.py`.
- **`lambda-envs/_probe_common.py`**: the env-agnostic half of the probe scripts, imported by both (`import _probe_common as common`). Holds the shared CLI (`add_common_args` + the strict `parse_args` that rewrites W&B's underscore spelling and rejects unknown flags, `apply_sweep_config`, `common_wandb_config`, `init_wandb`), the probe MLP (`init_probe_params`, `probe_forward`, `auroc`, `make_probe_trainer` → single-seed + vmapped drivers), the saturation-resistant metrics (`AGE_BUCKETS`, `cell_ages`, `probe_metrics_from_probs`/`probe_metrics`, `agg`), the multi-seed plumbing (`stack_states`, `unstack_state`, `split_each`, `episode_bounds`), FE checkpointing (`make_fe_checkpointer` → `--init-fe` / `--save-fe-every-eval`) and the two evaluators (`make_evaluate`, `make_evaluate_critic`). Each script supplies its env, its renderers and the two env-specific per-step callables the probe needs: **truth** (the hidden state to decode) and **observed** (which cells the memory has actually seen — the retention/inference split). It imports no env and no matplotlib, so `--vis-only` stays portable. Env-specific flags (board size, ship lengths, `--paper-arch`) stay in the env script. `--probe-rollout-policy` now takes `expert` as well as the old `bayes` spelling.
- **`lambda-envs/battleship_offline_bayes.py`** (superseded by the above, kept because HANDOFF.md results came from it): the memory question with the exploration loop cut out — a scripted **Bayes-density player** (count every ship placement consistent with the observed misses, weight `--hit-weight` per covered hit, fire the densest unfired cell; ε-greedy via `--offline-epsilon`, 0 = pure greedy) fills the buffer, then `--updates` gradient steps run with **zero env interaction**, then the probe runs along both the learned actor's and the Bayes player's rollouts (`--probe-policies`). It imports `battleship_board_probe.py` for the env/agent/probe/figures: that script is loaded with `--setup-only`, which builds its globals and `sys.exit`s before its Phase 1, and the importer catches the SystemExit — so every probe-script flag works here too. On 5×5 with ships `3,2` the player clears in ~13.3 shots (return ~12.7) vs ~21.7 random; the return scale is `rows*cols + 1 - shots`.
- **`lambda-envs/pocman_expert.py`**: scripted **privileged** PocMan expert (the counterpart of Battleship's Bayes player), used by `--expert-prefill-steps` / `--offline`. The maze is static, so all-pairs BFS distances precompute on host; the greedy base (nearest pellet by table distance, never step within `--expert-safety-margin` of a live ghost, chase ghosts while a power pill burns) is wrapped in one step of **rollout policy improvement** (simulate each first action + `--expert-rollout-depth` greedy steps in the real env, penalise death, argmax). Measured over 256 episodes: return **2890 ± 55**, 189/191 pellets, clears the board 48%, vs 164 / 15.2 pellets for uniform random. Rollout improvement is what makes it (greedy alone: 101 pellets, 97% deaths). Its signature is already `prefill_buffer`'s `behaviour_fn` contract; `epsilon=` adds coverage (ghosts are near-deterministic chasers, so episodes otherwise replay almost identically).
- **`lambda-envs/pocman_pellet_probe.py`**: the PocMan probe, mirroring `battleship_board_probe.py`'s interface through `_probe_common` (same flags, same offline / expert-prefill / PER / retrace / multi-seed / W&B behaviour). It probes the remaining-pellet map from the carry. **The retention/inference split does not apply here**: a pellet is gone *because* the agent ate it, so "hidden" and "observed" are complements and a fired/unfired split would leave one class per group — `probe_metrics_visitation` therefore reports decodability over all cells (`auroc`, `balanced`, `errors_per_state`, `exact_match`, `bits_per_cell`, `info_gain_bits`) and breaks down by **recency** (`bal_age*` = recall on cells eaten that long ago, `horizon_steps`). Unlike the Battleship script there is only one pipeline — the multi-seed one, used even for `--num-seeds 1`. `use_prev_action=True` is mandatory here: the observation is direction-symmetric, so only the action stream says which way the agent went.
- **`lambda-envs/battleship_board_probe.py`**: 4-phase probe analysis (train agent → collect hidden states + ground truth → train MLP probe → visualise) testing whether the recurrent memory encodes the hidden state. Battleship probes the ship board from the carry; because its raw obs carries no position, the probe agent feeds `[last_hit_miss, prev-action one-hot]` to the FE via an `ActionHistoryWrapper` (stores the last action in the state so it reaches `get_obs`), keeping the mask out. Phases are skippable (`--skip-train/-collect/-probe`); `--vis-only` renders from saved pkls with just jax+matplotlib. The probe accuracy is split into **fired cells = retention** (the memory directly observed them; tests long-term recall) vs **unfired cells = inference** (never observed; tests reconstructing the hidden board). **Use balanced accuracy / per-class recall, not raw accuracy** — ships are only ~14% of a 10×10 board, so "predict water everywhere" already scores ~86% and makes raw accuracy look flat/high regardless of memory. `board_probe_retention.png` plots per-class recall (hit→ship, miss→water) and balanced vs steps-since-fired — the retention horizon, which shrinks as `--memory-hidden-dim` is reduced. The episode/mp4 renders draw shots as **outlines** (not fills) so the probe's P(ship) belief stays visible under fired cells (filling them would just copy the ground-truth shot result into both panels).
