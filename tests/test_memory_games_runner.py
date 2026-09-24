"""CPU integration checks for the shared memory-games runner.

Scripted policies give independently countable evaluation trajectories; one tiny
real recurrent learner exercises replay, round boundaries and checkpoint resume.
"""

import argparse
import importlib
import json
from pathlib import Path
import pickle
from types import ModuleType, SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("gymnax")


@pytest.fixture(scope="module", autouse=True)
def cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@pytest.fixture(scope="module")
def runner():
    # Keep the canonical module names importable for pickled NamedTuple trees.
    examples = Path(__file__).resolve().parents[1] / "examples/lambda-envs"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(examples))
        return importlib.import_module("memory_games_sac")


def assert_tree_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(a, b)


def test_tracking_resume_keeps_run_and_updates_budget(runner, monkeypatch):
    """Resume one W&B record and extend its budget without a second init."""
    import sys

    calls = []
    updates = []
    metrics = []

    class Config(dict):
        def update(self, values, *, allow_val_change=False):
            updates.append(allow_val_change)
            super().update(values)

    run = SimpleNamespace(id="existing", name="original-name", config=Config(rounds=100))
    wb = ModuleType("wandb")
    wb.run = run

    def initialize(**kwargs):
        calls.append(kwargs)
        return run

    wb.init = initialize
    wb.define_metric = lambda *args, **kwargs: metrics.append((args, kwargs))
    monkeypatch.setitem(sys.modules, "wandb", wb)
    monkeypatch.setenv("WANDB_RUN_ID", "existing")
    monkeypatch.setenv("WANDB_RESUME", "must")
    args = SimpleNamespace(wandb=True, wandb_project="test-project",
                           wandb_run_name="original-name", num_seeds=3)
    tracking = runner.init_tracking(args, {"rounds": 200})
    assert tracking is wb
    assert calls == [{"project": "test-project"}]
    assert updates == [True]
    assert run.config["rounds"] == 200
    assert run.id == "existing" and run.name == "original-name"
    assert (("eval/*",), {"step_metric": "env_interactions"}) in metrics


def test_sweep_cli_accepts_underscores_but_rejects_unknown_flags(runner):
    parser = runner.build_parser()
    args = runner.parse_args(parser, ["--env=minesweeper", "--fe_lr=0.00002",
                                     "--num_seeds=5", "--final_return_window=3"])
    assert args.fe_lr == 2e-5 and args.num_seeds == 5
    assert args.final_return_window == 3
    with pytest.raises(SystemExit):
        runner.parse_args(parser, ["--env=minesweeper", "--misspelled_lr=0.01"])


def test_sweep_overrides_are_applied_and_outputs_are_isolated(runner, monkeypatch, tmp_path):
    import sys

    class Config(dict):
        def update(self, values, *, allow_val_change=False):
            assert allow_val_change
            super().update(values)

    wb = ModuleType("wandb")
    runs = []

    def initialize(**kwargs):
        run = SimpleNamespace(id=f"trial{len(runs)}", config=Config(
            env="minesweeper", method="ld", fe_lr=2e-5, num_seeds=5,
            remember=False, lambda_coef=0.04))
        runs.append(run)
        return run

    wb.init = initialize
    wb.define_metric = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "wandb", wb)
    monkeypatch.setenv("WANDB_SWEEP_ID", "sweep")
    monkeypatch.setenv("WANDB_RUN_ID", "controller-assigned")
    monkeypatch.setenv("WANDB_RESUME", "allow")
    monkeypatch.setenv("MEMORY_GAMES_OUTPUT_DIR", str(tmp_path))
    parser = runner.build_parser()
    outputs = []
    for i in range(2):
        args = runner.parse_args(parser, ["--env=minesweeper", "--fe-lr=0.0001"])
        attached = runner.prepare_sweep(parser, args)
        runner.resolve_args(parser, args)
        assert args.fe_lr == 2e-5 and args.num_seeds == 5
        assert args.method == "ld" and args.lambda_coef == 0.04 and args.wandb
        assert args.output_dir == tmp_path / "sweeps" / f"run_trial{i}"
        assert runner.init_tracking(args, runner.config_dict(args), attached) is wb
        assert len(runs) == i + 1  # tracking attaches to the controller's run
        outputs.append(args.output_dir)
    assert outputs[0] != outputs[1]


def test_sweep_rejects_resumed_trials_and_unknown_configuration(runner, monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_SWEEP_ID", "sweep")
    monkeypatch.setenv("MEMORY_GAMES_OUTPUT_DIR", str(tmp_path))
    parser = runner.build_parser()
    args = runner.parse_args(parser, ["--env=minesweeper", "--resume-from=checkpoint.pkl"])
    with pytest.raises(SystemExit):
        runner.prepare_sweep(parser, args)
    args = runner.parse_args(parser, ["--env=minesweeper"])
    monkeypatch.setattr(runner.common, "apply_sweep_config", lambda p, a:
                        SimpleNamespace(config={"misspelled_lr": 0.01}))
    with pytest.raises(SystemExit):
        runner.prepare_sweep(parser, args)


def test_sweep_files_match_budgets_and_shared_search_space(runner):
    yaml = pytest.importorskip("yaml")
    folder = Path(__file__).resolve().parents[1] / "examples/lambda-envs/sweeps"
    configs = [yaml.safe_load((folder / f"minesweeper_{method}.yaml").read_text())
               for method in ("baseline", "ld")]
    baseline, ld = [config["parameters"] for config in configs]
    shared = set(baseline) - {"method", "lambda_coef"}
    assert set(baseline) == set(ld)
    assert {k: baseline[k] for k in shared} == {k: ld[k] for k in shared}
    assert baseline["method"] == {"value": "sac"}
    assert ld["method"] == {"value": "ld"}
    assert baseline["lambda_coef"] == {"value": 0.0}
    assert ld["lambda_coef"]["min"] > 0
    for config in configs:
        assert config["metric"] == {"name": "final/return_smoothed/mean", "goal": "maximize"}
        assert "early_terminate" not in config
        argv = ["--wandb"]
        for name, setting in config["parameters"].items():
            value = setting.get("value", setting.get("min"))
            if isinstance(value, bool):
                if value:
                    argv.append(f"--{name}")
            else:
                argv.append(f"--{name}={value}")
        parser = runner.build_parser()
        args = runner.parse_args(parser, argv)
        runner.resolve_args(parser, args)
        assert args.rounds * args.train_steps == 200000
        assert args.num_seeds == 5 and args.prefill_steps == 3008
        assert args.final_return_window == 5 and args.eval_every == 1
        assert args.eval_episodes == 128
        assert args.burn_in_length == 30 and not args.remember
        assert args.checkpoint_every == args.rounds


def fixed_env(runner, game, remember=False):
    """Real game rules and gymnax auto-reset, with tiny reproducible boards."""
    if game == "concentration":
        class FixedConcentration(runner.Concentration):
            def reset_env(self, key, params):
                _, state = super().reset_env(key, params)
                state = state._replace(cards=jnp.array([0, 1, 0, 1], jnp.int32))
                return self.get_obs(state, params), state

        base = FixedConcentration(4, 2, remember=remember)
    else:
        class FixedMinesweeper(runner.MineSweeper):
            def reset_env(self, key, params):
                _, state = super().reset_env(key, params)
                state = state._replace(
                    mines=jnp.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], bool),
                    neighbor_counts=jnp.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]], jnp.int32),
                )
                return self.get_obs(state, params), state

        base = FixedMinesweeper(3, 3, 1, remember=remember)
    return runner.ActionHistoryEnv(base)


@pytest.mark.parametrize("game", ["concentration", "minesweeper"])
@pytest.mark.parametrize("remember", [False, True])
def test_action_history_records_executed_actions_even_when_repeated(runner, game, remember):
    env = fixed_env(runner, game, remember)
    key = jax.random.key(7)
    obs, state = env.reset(key)
    base_obs, inner = env.env.reset(key)
    assert obs.shape == env.observation_space().shape
    assert obs.dtype == jnp.float32
    assert env.action_space().n == env.num_actions
    np.testing.assert_array_equal(obs[:-env.num_actions], base_obs)
    np.testing.assert_array_equal(obs[-env.num_actions:], 0)

    # These are the actions actually executed, including a penalized duplicate.
    # A wrapper must not retain the last *successful* action instead.
    step = jax.jit(env.step)
    for action in [1, 1, 0]:
        base_obs, inner, reward, done, info = env.env.step(key, inner, action)
        obs, state, wrapped_reward, wrapped_done, wrapped_info = step(
            key, state, jnp.int32(action))
        assert not done and not wrapped_done
        assert_tree_equal((state.inner, wrapped_reward, wrapped_info), (inner, reward, info))
        np.testing.assert_array_equal(obs[:-env.num_actions], base_obs)
        np.testing.assert_array_equal(state.previous_action, np.eye(env.num_actions)[action])
        np.testing.assert_array_equal(obs[-env.num_actions:], state.previous_action)
        np.testing.assert_array_equal(obs, env.get_obs(state))
        assert_tree_equal(env.diagnostics(state, action), env.env.diagnostics(inner, action))


@pytest.mark.parametrize("game", ["concentration", "minesweeper"])
def test_action_history_mixed_batch_autoresets_exactly_once(runner, game):
    # Use ordinary randomized resets: comparing the complete reset state also
    # catches an erroneous second reset with a different random key.
    base = runner.Concentration(4, 2) if game == "concentration" else runner.MineSweeper(3, 3, 1)
    env = runner.ActionHistoryEnv(base)
    _, initial = env.reset(jax.random.key(2))
    action = 0 if game == "concentration" else int(jnp.argmin(initial.inner.mines))
    timeout = initial._replace(inner=initial.inner._replace(
        timestep=jnp.int32(env.episode_length - 1)))
    states = jax.tree.map(lambda a, b: jnp.stack([a, b]), initial, timeout)
    keys = jax.random.split(jax.random.key(31), 2)
    obs, states, rewards, dones, info = jax.jit(jax.vmap(env.step))(
        keys, states, jnp.full(2, action, jnp.int32))
    np.testing.assert_array_equal(dones, [False, True])
    np.testing.assert_array_equal(states.previous_action[0], np.eye(env.num_actions)[action])
    np.testing.assert_array_equal(states.previous_action[1], 0)
    reset_obs, reset = env.reset(jax.random.split(keys[1])[1])
    assert_tree_equal((obs[1], jax.tree.map(lambda x: x[1], states)), (reset_obs, reset))
    _, _, reward, done, terminal_info = base.step(keys[1], timeout.inner, action)
    assert done
    assert_tree_equal((rewards[1], jax.tree.map(lambda x: x[1], info)), (reward, terminal_info))
    np.testing.assert_array_equal(obs, jax.vmap(env.get_obs)(states))

    # Explicit gymnax params must also reach the underlying environment.
    params = env.default_params._replace(max_steps_in_episode=1)
    obs, reset, _, done, _ = env.step(keys[0], initial, action, params)
    assert done and int(reset.inner.timestep) == 0
    np.testing.assert_array_equal(obs[-env.num_actions:], 0)


@pytest.mark.parametrize("actions,candidates", [
    ([], [0, 1, 2, 3]),
    ([0, 1], [2, 3]),                 # explore after a mismatch
    ([0, 1, 2], [0]),                # remembered partner, excluding the first card
    ([0, 1, 2, 3], [0, 1, 2, 3]),    # start either known pair
    ([0, 1, 2, 0], [3]),             # matched positions are no longer available
])
def test_concentration_history_policy_uses_only_revealed_ranks(runner, actions, candidates):
    env = fixed_env(runner, "concentration")
    _, state = env.reset(jax.random.key(0))
    for action in actions:
        _, state, _, done, _ = env.step(jax.random.key(0), state, action)
        assert not done
    # Poison both private ranks and the unused entries of the history table.
    changed = state._replace(inner=state.inner._replace(
        cards=jnp.full_like(state.inner.cards, -999),
        last_seen=jnp.where(state.inner.seen, state.inner.last_seen, 0)))
    choose = jax.jit(jax.vmap(lambda s, k: runner.history_action(env, s, k), in_axes=(None, 0)))
    keys = jax.random.split(jax.random.key(71), 64)
    actual = choose(state, keys)
    np.testing.assert_array_equal(actual, choose(changed, keys))
    assert np.isin(actual, candidates).all()
    if len(candidates) > 1:
        assert np.unique(actual).size > 1


@pytest.mark.parametrize("actions,candidates", [
    ([], list(range(9))),
    ([0], [1, 3, 4]),                # only neighbors of the revealed corner zero
    ([4], [0, 1, 2, 3, 5, 6, 7, 8]),  # an observed nonzero cannot reveal hidden zeros
    ([0, 1, 3, 4], [2, 5, 6, 7]),
])
def test_minesweeper_history_policy_ignores_mines_and_unseen_clues(runner, actions, candidates):
    env = fixed_env(runner, "minesweeper")
    _, state = env.reset(jax.random.key(0))
    for action in actions:
        _, state, _, done, _ = env.step(jax.random.key(0), state, action)
        assert not done
    changed = state._replace(inner=state.inner._replace(
        mines=~state.inner.mines,
        neighbor_counts=jnp.where(state.inner.viewed, state.inner.neighbor_counts, 0)))
    choose = jax.jit(jax.vmap(lambda s, k: runner.history_action(env, s, k), in_axes=(None, 0)))
    keys = jax.random.split(jax.random.key(72), 64)
    actual = choose(state, keys)
    np.testing.assert_array_equal(actual, choose(changed, keys))
    assert np.isin(actual, candidates).all()
    # With no observed zero, even a mine is a legitimate unknown candidate.
    if actions in ([], [4]):
        assert 8 in np.asarray(actual)


@pytest.mark.parametrize("mode", ["greedy", "sampled"])
@pytest.mark.parametrize("game", ["concentration", "minesweeper"])
def test_evaluator_counts_terminal_reward_progress_and_invalid_actions(runner, game, mode):
    env = fixed_env(runner, game)

    def predict(script, obs, carry, key, *, deterministic):
        assert deterministic is (mode == "greedy")
        return script[carry[0].astype(jnp.int32)], carry + 1

    if game == "concentration":
        # First seed: mismatch, then two remembered matches; stops at step 6.
        # Second seed: four invalid second flips, timing out without a match.
        scripts = [[0, 1, 2, 0, 3, 1, 0, 0], [0, 0, 1, 1, 2, 2, 3, 3]]
        expected = {"return": [0.75, -1], "steps": [6, 8], "progress": [1, 0],
                    "success": [1, 0], "opportunities": [2, 2],
                    "known_choice_rate": [1, 0], "invalid_fraction": [0, 0.5]}
    else:
        # First seed clears eight cells. Second reveals one, repeats, then dies;
        # third dies immediately. Later auto-reset episodes must contribute zero.
        scripts = [list(range(8)), [0, 0, 8, 0, 1, 2, 3, 4], [8, 0, 1, 2, 3, 4, 5, 6]]
        expected = {"return": [1, -7 / 12, -5 / 8], "steps": [8, 3, 1],
                    "progress": [1, 1 / 8, 0], "success": [1, 0, 0],
                    "opportunities": [7, 2, 0], "known_choice_rate": [1, 0, np.nan],
                    "invalid_fraction": [0, 1 / 3, 0]}
    evaluate = runner.make_evaluator(SimpleNamespace(predict=predict), env, 1, 3, mode)
    actual = evaluate(jnp.asarray(scripts, jnp.float32), jax.random.split(jax.random.key(8), len(scripts)))
    assert actual.keys() == expected.keys()
    for name, values in expected.items():
        np.testing.assert_allclose(actual[name], values, atol=1e-6, err_msg=name)


class DiagnosticState(NamedTuple):
    kind: jax.Array
    time: jax.Array


def test_evaluator_pools_opportunities_and_steps_instead_of_episode_rates(runner):
    # Three episode types have (opportunities, taken) = (0,0), (1,1), (3,1)
    # and lengths 1, 2, 4. A ratio of sums differs from a mean of ratios.
    rewards = jnp.array([[-0.5, 0, 0, 0], [0.25, -0.75, 0, 0],
                         [0.25, -0.1, 0.25, -0.75], [10, 10, 10, 10]], jnp.float32)
    opportunities = jnp.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [1, 1, 1, 1]])
    taken = jnp.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 1, 1]])
    invalid = jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1]])

    class DiagnosticEnv:
        env = runner.MineSweeper(3, 3, 1)  # selects the Minesweeper metric names
        default_params = None
        episode_length = 4
        num_actions = 9

        def reset(self, key, params=None):
            state = DiagnosticState(jax.random.randint(key, (), 0, 3), jnp.int32(0))
            return jnp.zeros(1), state

        def diagnostics(self, state, action):
            index = (state.kind, state.time)
            return {"known_safe_opportunity": opportunities[index],
                    "known_safe_taken": taken[index], "repeat_action": invalid[index]}

        def step(self, key, state, action, params=None):
            reward = rewards[state.kind, state.time]
            done = state.time + 1 >= jnp.array([1, 2, 4, 4])[state.kind]
            # Auto-reset into a lucrative episode: any post-terminal leakage
            # pollutes reward, progress AND both diagnostic denominators.
            state = DiagnosticState(jnp.where(done, 3, state.kind),
                                    jnp.where(done, 0, state.time + 1))
            return jnp.zeros(1), state, reward, done, {}

    env = DiagnosticEnv()
    fns = SimpleNamespace(predict=lambda agent, obs, memory, key, **kw: (jnp.float32(0), memory))
    keys = jax.random.split(jax.random.key(109), 2)
    actual = runner.make_evaluator(fns, env, 0, 32, "greedy")(jnp.zeros(2), keys)
    for seed, key in enumerate(keys):
        episode_keys = jax.random.split(key, 32)
        kinds = [int(env.reset(jax.random.split(k)[1])[1].kind) for k in episode_keys]
        n0, n1, n2 = np.bincount(kinds, minlength=3)
        assert min(n0, n1, n2) > 0
        rate = (n1 + n2) / (n1 + 3 * n2)
        assert not np.isclose(rate, (n1 + n2 / 3) / (n1 + n2))
        expected = {"return": (-0.5 * n0 - 0.5 * n1 - 0.35 * n2) / 32,
                    "steps": (n0 + 2 * n1 + 4 * n2) / 32,
                    "progress": (n1 + 2 * n2) / (32 * 4), "success": 0,
                    "opportunities": (n1 + 3 * n2) / 32,
                    "known_choice_rate": rate,
                    "invalid_fraction": n2 / (n0 + 2 * n1 + 4 * n2)}
        for name, value in expected.items():
            assert float(actual[name][seed]) == pytest.approx(value, abs=1e-6), name


@pytest.mark.parametrize("mode", ["random", "history"])
def test_reference_evaluators_do_not_call_the_actor(runner, mode):
    env = runner.ActionHistoryEnv(runner.Concentration(2, 1))

    def forbidden(*args, **kwargs):
        raise AssertionError("reference evaluation must not use the actor")

    result = runner.make_evaluator(SimpleNamespace(predict=forbidden), env, 0, 16, mode)(
        jnp.zeros(1), jax.random.split(jax.random.key(5), 1))
    assert 2 <= float(result["steps"][0]) <= 4
    assert 0 <= float(result["progress"][0]) <= 1
    if mode == "history":
        assert float(result["return"][0]) == float(result["success"][0]) == 1
        assert float(result["steps"][0]) == 2
        assert np.isnan(result["known_choice_rate"][0])  # second card was still unseen


@pytest.fixture(scope="module")
def tiny_run(runner, tmp_path_factory):
    p = runner.build_parser()
    args = p.parse_args([
        "--env", "concentration", "--cards", "4", "--types", "2", "--rounds", "2",
        "--train-steps", "1", "--num-seeds", "2", "--seed", "37", "--method", "ld",
        "--memory-type", "rnn", "--memory-hidden-dim", "2", "--projection-dim", "3",
        "--head-dim", "3", "--batch-size", "1", "--sequence-length", "1",
        "--burn-in-length", "1", "--lambda-truncation", "1", "--online-buffer-size", "16",
        "--prefill-steps", "4", "--eval-episodes", "2",
        "--output-dir", str(tmp_path_factory.mktemp("memory-games")),
    ])
    env = runner.resolve_args(p, args)
    spec = runner.env_spec_from_gymnax(env, env.default_params)
    hp = runner.Hyperparameters(
        batch_size=1, online_buffer_size=16, burn_in_length=1, sequence_length=1,
        lambda_truncation=1, retrace=True, fake_onpolicy_loss=False,
        lambda1=0, lambda2=0.95, lambda_coef=0.01, alpha=args.alpha,
        stop_actor_fe=True)
    agent, fns = runner.create_iqlearn_from_env(
        spec, {"observations": jnp.zeros((1, *spec.obs_shape)), "actions": jnp.zeros((1, 1))},
        buffer_size=1, hp=hp, projection=3, memory_type="rnn", memory_hidden_dim=2,
        use_prev_action=False, actor_dims=(3,), critic_dims=(3,),
        lambda1_critic_dims=(3,), lambda2_critic_dims=(3,), approximate_lambda=True,
        critic_layer_norm=True, train_steps=1, seed=args.seed, use_sac=args.use_sac)
    _, es = env.reset(jax.random.key(40))
    agent, _ = fns.prefill_buffer(agent, env, env.default_params, es, 4, jax.random.key(41))
    # Match main(): prefill's artificial terminal must be followed by a real reset.
    _, es = env.reset(jax.random.key(42))
    memory = jnp.zeros(2, jnp.float32)
    train = jax.jit(lambda s, es, c, k: fns.train_unrolled(s, env, env.default_params, es, c, k))
    snapshots = [(agent, es, memory)]
    for seed in (43, 44):
        agent, es, memory, metrics = train(agent, es, memory, jax.random.key(seed))
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(metrics))
        snapshots.append((agent, es, memory))
    return SimpleNamespace(args=args, env=env, fns=fns, train=train, snapshots=snapshots)


def test_real_recurrent_training_preserves_both_histories_across_rounds(tiny_run):
    env, fns = tiny_run.env, tiny_run.fns
    for before, after in zip(tiny_run.snapshots, tiny_run.snapshots[1:]):
        agent, es, memory = before
        next_agent, next_es, next_memory = after
        obs = env.get_obs(es)
        _, expected_memory = fns.predict(agent, obs, memory, jax.random.key(90))
        np.testing.assert_allclose(next_memory, expected_memory, atol=1e-6)
        slot = int(agent.online_buffer.pos)
        replay = next_agent.online_buffer.info
        np.testing.assert_array_equal(replay["observations"][slot], obs)
        action = int(replay["actions"][slot, 0])
        np.testing.assert_array_equal(next_es.previous_action, np.eye(env.num_actions)[action])
        assert not replay["terminated"][slot]
    agent, es, memory = tiny_run.snapshots[1]
    assert np.any(memory != 0) and np.any(es.previous_action != 0)
    _, lost_memory = fns.predict(agent, env.get_obs(es), jnp.zeros_like(memory), jax.random.key(90))
    assert not np.allclose(tiny_run.snapshots[2][2], lost_memory)

    # A true episode end resets both histories, even in the first step of a round.
    agent, es, memory = tiny_run.snapshots[2]
    es = es._replace(inner=es.inner._replace(timestep=jnp.int32(env.episode_length - 1)))
    next_agent, es, memory, _ = tiny_run.train(agent, es, memory, jax.random.key(45))
    assert next_agent.online_buffer.info["terminated"][int(agent.online_buffer.pos)]
    np.testing.assert_array_equal(memory, 0)
    np.testing.assert_array_equal(es.previous_action, 0)
    assert int(es.inner.timestep) == 0


@pytest.fixture
def checkpoint(runner, tiny_run, tmp_path):
    agents, states, carries = jax.tree.map(lambda a, b: jnp.stack([a, b]),
                                         *tiny_run.snapshots[1:])
    keys = jax.random.split(jax.random.PRNGKey(98), 2)
    history = [{"round": 1, "returns": [0.25, -0.5]}, {"round": 2, "returns": [0.5, 0.75]}]
    path = tmp_path / "checkpoint.pkl"
    runner.save_checkpoint(path, tiny_run.args, agents, states, carries, keys, 2, history)
    assert path.exists() and not path.with_suffix(".tmp").exists()
    return path, (agents, states, carries, keys), history


def test_checkpoint_round_trip_and_next_update_match_uninterrupted_training(runner, tiny_run, checkpoint):
    path, expected, history = checkpoint
    args = argparse.Namespace(**vars(tiny_run.args))
    args.rounds = 5
    args.eval_every, args.eval_episodes, args.checkpoint_every = 2, 3, 0
    args.output_dir, args.resume_from = path.parent / "resumed", path
    args.wandb, args.wandb_project, args.wandb_run_name = True, "test-project", "resumed"
    agents, states, carries, keys, rnd, restored_history = runner.load_checkpoint(path, args)
    assert rnd == 2 and restored_history == history
    assert_tree_equal((agents, states, carries, keys), expected)
    assert set(agents.online_buffer.info) >= {
        "observations", "actions", "rewards", "terminated", "behaviour_weight"}
    assert np.all(agents.update_step > 0)
    assert np.all(agents.online_buffer.pos > args.prefill_steps)
    assert_tree_equal(runner.common.split_each(keys), runner.common.split_each(expected[3]))
    restored = jax.tree.map(lambda x: x[1], (agents, states, carries, keys))
    uninterrupted = jax.tree.map(lambda x: x[1], expected)
    assert_tree_equal(tiny_run.train(*restored), tiny_run.train(*uninterrupted))


@pytest.mark.parametrize("field,value", [
    ("env", "minesweeper"), ("method", "critics"), ("seed", 99), ("num_seeds", 3),
    ("train_steps", 2), ("memory_type", "lstm"), ("projection_dim", 9),
    ("sequence_length", 3), ("burn_in_length", 2), ("online_buffer_size", 32),
    ("remember", True), ("gamma", 0.9), ("lambda_coef", 0.2), ("uncorrected", True),
])
def test_checkpoint_rejects_immutable_training_config_changes(runner, tiny_run, checkpoint, field, value):
    args = argparse.Namespace(**vars(tiny_run.args))
    setattr(args, field, value)
    with pytest.raises(ValueError, match=field):
        runner.load_checkpoint(checkpoint[0], args)


def test_checkpoint_requires_supported_version_and_total_round_budget(runner, tiny_run, checkpoint):
    path = checkpoint[0]
    args = argparse.Namespace(**vars(tiny_run.args))
    assert runner.load_checkpoint(path, args)[-2] == 2  # equal total is valid
    args.rounds = 1
    with pytest.raises(ValueError, match="TOTAL.*smaller"):
        runner.load_checkpoint(path, args)
    with path.open("rb") as stream:
        snapshot = pickle.load(stream)
    snapshot["version"] = 999
    with path.open("wb") as stream:
        pickle.dump(snapshot, stream)
    with pytest.raises(ValueError, match="version"):
        runner.load_checkpoint(path, tiny_run.args)


def test_main_threads_history_through_rounds_and_resume(runner, monkeypatch, tmp_path):
    """Exercise the real compiled/vmapped host loop using an inspectable learner.

    Its memory counts observations, and its trace records every round's input.
    The real recurrent/optimizer path is covered above; here a discarded carry
    or a repeated prefill on resume is visible without relying on learned weights.
    """
    def build(spec, expert, **kwargs):
        assert kwargs["use_prev_action"] is False  # history is already in obs
        memory_dim = kwargs["memory_hidden_dim"]
        agent = {"updates": jnp.int32(0), "prefills": jnp.int32(0),
                 "observations": jnp.zeros((3, *spec.obs_shape)),
                 "memories": jnp.zeros((3, memory_dim)), "times": jnp.zeros(3, jnp.int32)}

        def predict(agent, obs, memory, key, **kwargs):
            return jnp.float32(1), memory + 1

        def prefill(agent, env, params, es, n_steps, key):
            # Deliberately leave a live episode with nonzero action history.
            # main must reset it because real prefill marks a synthetic terminal.
            _, es, _, _, _ = env.step(key, es, jnp.int32(1), params)
            return {**agent, "prefills": agent["prefills"] + 1}, es

        def train(agent, env, params, es, memory, key):
            index = agent["updates"]
            obs = env.get_obs(es, params)
            agent = {**agent, "updates": index + 1,
                     "observations": agent["observations"].at[index].set(obs),
                     "memories": agent["memories"].at[index].set(memory),
                     "times": agent["times"].at[index].set(es.inner.timestep)}
            _, es, _, done, _ = env.step(key, es, jnp.int32(1), params)
            return agent, es, jnp.where(done, 0, memory + 1), {"critic_loss": jnp.float32(0)}

        return agent, SimpleNamespace(predict=predict, prefill_buffer=prefill, train_unrolled=train)

    monkeypatch.setattr(runner, "create_iqlearn_from_env", build)
    common = ["--env", "concentration", "--cards", "4", "--types", "2",
              "--train-steps", "1", "--num-seeds", "2", "--seed", "41",
              "--memory-type", "rnn", "--memory-hidden-dim", "2", "--projection-dim", "3",
              "--head-dim", "3", "--batch-size", "1", "--sequence-length", "1",
              "--burn-in-length", "1", "--lambda-truncation", "1",
              "--online-buffer-size", "16", "--prefill-steps", "4",
              "--eval-episodes", "2", "--checkpoint-every", "1", "--final-return-window", "2"]
    resumed_dir, uninterrupted_dir = tmp_path / "resumed", tmp_path / "uninterrupted"
    runner.main(common + ["--rounds", "2", "--output-dir", str(resumed_dir)])
    resume_argv = common + ["--rounds", "3", "--output-dir", str(resumed_dir),
                            "--resume-from", str(resumed_dir / "checkpoint.pkl")]
    runner.main(resume_argv)
    runner.main(common + ["--rounds", "3", "--output-dir", str(uninterrupted_dir)])

    parser = runner.build_parser()
    args = parser.parse_args(resume_argv)
    runner.resolve_args(parser, args)
    restored = runner.load_checkpoint(resumed_dir / "checkpoint.pkl", args)
    uninterrupted = runner.load_checkpoint(uninterrupted_dir / "checkpoint.pkl", args)
    assert_tree_equal(restored[:4], uninterrupted[:4])
    assert restored[4:] == uninterrupted[4:]
    agent, es, memory, _, rnd, history = restored
    assert rnd == 3 and [row["round"] for row in history] == [1, 2, 3]
    np.testing.assert_array_equal(agent["prefills"], [1, 1])
    np.testing.assert_array_equal(agent["updates"], [3, 3])
    np.testing.assert_array_equal(agent["times"], [[0, 1, 2], [0, 1, 2]])
    np.testing.assert_array_equal(agent["memories"], np.broadcast_to(
        np.arange(3)[None, :, None], (2, 3, 2)))
    expected_pa = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]])
    np.testing.assert_array_equal(agent["observations"][:, :, -4:],
                                  np.broadcast_to(expected_pa, (2, 3, 4)))
    np.testing.assert_array_equal(memory, 3)
    np.testing.assert_array_equal(es.inner.timestep, [3, 3])
    np.testing.assert_array_equal(es.previous_action, [[0, 1, 0, 0], [0, 1, 0, 0]])

    def reject_non_json_constant(value):
        raise AssertionError(f"nonstandard JSON constant: {value}")

    rows = [json.loads(line, parse_constant=reject_non_json_constant)
            for line in (resumed_dir / "metrics.jsonl").read_text().splitlines()]
    assert [row["round"] for row in rows] == [0, 0, 1, 2, 3]
    assert [row["env_interactions"] for row in rows] == [4, 4, 5, 6, 7]
    assert rows[-1]["eval/greedy/known_choice_rate/mean"] is None
    summary = json.loads((resumed_dir / "summary.json").read_text())
    assert summary["round"] == summary["train_env_steps"] == 3
    assert summary["env_interactions"] == 7
    assert summary["final/evaluations_averaged"] == 2
    expected = np.asarray([row["returns"] for row in history[-2:]]).mean(axis=0).mean()
    assert summary["final/return_smoothed/mean"] == pytest.approx(expected)
    assert summary["final/window_start_train_env_steps"] == 2
    assert summary["final/window_end_train_env_steps"] == 3
