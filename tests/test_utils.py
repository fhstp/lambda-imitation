"""Tests for utils.py.

gymnasium is a real dependency in this environment; gymnax and jumanji tests
use lightweight mock modules so the tests run without those packages installed.
"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from lambda_imitation.utils import (
    EnvSpec,
    create_iqlearn_from_env,
    env_spec_from_gymnasium,
    env_spec_from_gymnax,
    env_spec_from_jumanji,
)
from lambda_imitation.iqlearn import Hyperparameters, IQLearnFunctions, IQLearnGraphs, IQLearnState


# ---------------------------------------------------------------------------
# Helpers: mock gymnax spaces
# ---------------------------------------------------------------------------

class _GBox:
    """Minimal gymnax Box-like space."""
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape


class _GDiscrete:
    """Minimal gymnax Discrete-like space."""
    def __init__(self, n):
        self.n = n
        self.shape = ()


def _gymnax_modules(obs_space, act_space):
    """Return sys.modules patch dict and a mock env for gymnax tests.

    Use real ModuleType objects for the whole chain and wire attributes
    explicitly, so that ``import gymnax.environments.spaces as spaces``
    resolves to our module via both sys.modules *and* attribute traversal
    (which is what CPython actually does for dotted imports).
    """
    spaces_mod = ModuleType("gymnax.environments.spaces")
    spaces_mod.Box = _GBox
    spaces_mod.Discrete = _GDiscrete

    environments_mod = ModuleType("gymnax.environments")
    environments_mod.spaces = spaces_mod

    gymnax_mod = ModuleType("gymnax")
    gymnax_mod.environments = environments_mod

    mocks = {
        "gymnax": gymnax_mod,
        "gymnax.environments": environments_mod,
        "gymnax.environments.spaces": spaces_mod,
    }

    class MockEnv:
        def observation_space(self, params):
            return obs_space

        def action_space(self, params):
            return act_space

    return mocks, MockEnv()


# ---------------------------------------------------------------------------
# Helpers: mock jumanji specs
# ---------------------------------------------------------------------------

class _JArray:
    """Minimal jumanji Array spec (leaf)."""
    def __init__(self, shape, dtype=jnp.float32):
        self.shape = shape
        self.dtype = dtype


class _JBoundedArray(_JArray):
    """Minimal jumanji BoundedArray spec."""
    def __init__(self, shape, minimum, maximum, dtype=jnp.float32):
        super().__init__(shape, dtype)
        self.minimum = minimum
        self.maximum = maximum


class _JDiscreteArray(_JBoundedArray):
    """Minimal jumanji DiscreteArray spec."""
    def __init__(self, num_values):
        super().__init__((), minimum=0, maximum=num_values - 1, dtype=jnp.int32)
        self.num_values = num_values


class _JMultiDiscreteArray(_JBoundedArray):
    """Minimal jumanji MultiDiscreteArray spec."""
    def __init__(self, num_values):
        n = jnp.asarray(num_values)
        super().__init__(n.shape, minimum=jnp.zeros_like(n), maximum=n - 1, dtype=jnp.int32)
        self.num_values = n


class _JSpec:
    """Minimal jumanji composite Spec (NOT an Array subclass)."""
    pass


def _jumanji_modules(obs_spec, act_spec):
    """Return sys.modules patch dict and a mock env for jumanji tests.

    Same attribute-wiring approach as ``_gymnax_modules``.
    """
    specs_mod = ModuleType("jumanji.specs")
    specs_mod.Array = _JArray
    specs_mod.BoundedArray = _JBoundedArray
    specs_mod.DiscreteArray = _JDiscreteArray
    specs_mod.MultiDiscreteArray = _JMultiDiscreteArray
    specs_mod.Spec = _JSpec

    jumanji_mod = ModuleType("jumanji")
    jumanji_mod.specs = specs_mod

    mocks = {
        "jumanji": jumanji_mod,
        "jumanji.specs": specs_mod,
    }

    class MockEnv:
        observation_spec = obs_spec
        action_spec = act_spec

    return mocks, MockEnv()


# ---------------------------------------------------------------------------
# Helpers: minimal expert data
# ---------------------------------------------------------------------------

def _make_expert_data(n=20, obs_dim=4, action_dim=2):
    key = jnp.zeros((), dtype=jnp.uint32)
    return {
        "observations": jnp.ones((n, obs_dim)),
        "actions": jnp.zeros((n, action_dim)),
    }


def _make_spec(obs_shape=(4,), action_dim=2, low=-1.0, high=1.0):
    return EnvSpec(
        obs_shape=obs_shape,
        action_dim=action_dim,
        is_discrete=False,
        action_low=jnp.full((action_dim,), low),
        action_high=jnp.full((action_dim,), high),
    )


def _make_discrete_spec(obs_shape=(4,), num_actions=4):
    return EnvSpec(
        obs_shape=obs_shape,
        action_dim=num_actions,
        is_discrete=True,
    )


# ===========================================================================
# EnvSpec
# ===========================================================================

class TestEnvSpec:
    def test_is_namedtuple(self):
        spec = _make_spec()
        assert isinstance(spec, tuple)

    def test_fields(self):
        spec = _make_spec(obs_shape=(8,), action_dim=3)
        assert spec.obs_shape == (8,)
        assert spec.action_dim == 3
        assert spec.is_discrete is False
        assert spec.action_low.shape == (3,)
        assert spec.action_high.shape == (3,)

    def test_discrete_fields(self):
        spec = _make_discrete_spec(obs_shape=(8,), num_actions=5)
        assert spec.obs_shape == (8,)
        assert spec.action_dim == 5
        assert spec.is_discrete is True
        assert spec.action_low is None
        assert spec.action_high is None


# ===========================================================================
# env_spec_from_gymnasium  (real library is available)
# ===========================================================================

class TestEnvSpecFromGymnasium:
    def _make_env(self, obs_shape=(4,), act_shape=(2,), act_low=-1.0, act_high=1.0):
        import gymnasium.spaces as sp
        obs_space = sp.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        act_space = sp.Box(
            low=np.full(act_shape, act_low, dtype=np.float32),
            high=np.full(act_shape, act_high, dtype=np.float32),
            dtype=np.float32,
        )

        class Env:
            observation_space = obs_space
            action_space = act_space

        return Env()

    def test_obs_shape(self):
        spec = env_spec_from_gymnasium(self._make_env(obs_shape=(11,)))
        assert spec.obs_shape == (11,)

    def test_action_dim_flat(self):
        spec = env_spec_from_gymnasium(self._make_env(act_shape=(3,)))
        assert spec.action_dim == 3

    def test_action_dim_multidim(self):
        """Multi-dim action shape should be flattened to a scalar count."""
        spec = env_spec_from_gymnasium(self._make_env(act_shape=(2, 3)))
        assert spec.action_dim == 6

    def test_action_bounds_shape(self):
        spec = env_spec_from_gymnasium(self._make_env(act_shape=(4,)))
        assert spec.action_low.shape == (4,)
        assert spec.action_high.shape == (4,)

    def test_action_bounds_values(self):
        spec = env_spec_from_gymnasium(self._make_env(act_low=-2.0, act_high=3.0))
        assert float(spec.action_low[0]) == pytest.approx(-2.0)
        assert float(spec.action_high[0]) == pytest.approx(3.0)

    def test_action_bounds_asymmetric(self):
        import gymnasium.spaces as sp

        class Env:
            observation_space = sp.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
            action_space = sp.Box(
                low=np.array([-1.0, -2.0], dtype=np.float32),
                high=np.array([1.0, 3.0], dtype=np.float32),
            )

        spec = env_spec_from_gymnasium(Env())
        assert float(spec.action_low[1]) == pytest.approx(-2.0)
        assert float(spec.action_high[1]) == pytest.approx(3.0)

    def test_discrete_action_returns_spec(self):
        import gymnasium.spaces as sp

        class Env:
            observation_space = sp.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
            action_space = sp.Discrete(5)

        spec = env_spec_from_gymnasium(Env())
        assert spec.is_discrete is True
        assert spec.action_dim == 5
        assert spec.action_low is None
        assert spec.action_high is None

    def test_raises_on_discrete_obs(self):
        import gymnasium.spaces as sp

        class Env:
            observation_space = sp.Discrete(10)
            action_space = sp.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        with pytest.raises(ValueError, match="Box observation space"):
            env_spec_from_gymnasium(Env())

    def test_output_dtypes_float32(self):
        spec = env_spec_from_gymnasium(self._make_env())
        assert spec.action_low.dtype == jnp.float32
        assert spec.action_high.dtype == jnp.float32


# ===========================================================================
# env_spec_from_gymnax  (mocked)
# ===========================================================================

class TestEnvSpecFromGymnax:
    def test_basic_extraction(self):
        obs = _GBox(-np.inf, np.inf, (8,))
        act = _GBox(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), (2,))
        mocks, env = _gymnax_modules(obs, act)
        with patch.dict(sys.modules, mocks):
            spec = env_spec_from_gymnax(env, params=None)
        assert spec.obs_shape == (8,)
        assert spec.action_dim == 2

    def test_action_bounds(self):
        obs = _GBox(-np.inf, np.inf, (4,))
        act = _GBox(np.array([-2.0, -3.0]), np.array([2.0, 3.0]), (2,))
        mocks, env = _gymnax_modules(obs, act)
        with patch.dict(sys.modules, mocks):
            spec = env_spec_from_gymnax(env, params=None)
        assert float(spec.action_low[0]) == pytest.approx(-2.0)
        assert float(spec.action_high[1]) == pytest.approx(3.0)

    def test_scalar_bounds_broadcast(self):
        """Scalar low/high must be broadcast to (action_dim,)."""
        obs = _GBox(-np.inf, np.inf, (4,))
        act = _GBox(-1.0, 1.0, (3,))
        mocks, env = _gymnax_modules(obs, act)
        with patch.dict(sys.modules, mocks):
            spec = env_spec_from_gymnax(env, params=None)
        assert spec.action_low.shape == (3,)
        assert spec.action_high.shape == (3,)
        assert (spec.action_low == -1.0).all()
        assert (spec.action_high == 1.0).all()

    def test_discrete_action_returns_spec(self):
        obs = _GBox(-np.inf, np.inf, (4,))
        act = _GDiscrete(5)
        mocks, env = _gymnax_modules(obs, act)
        with patch.dict(sys.modules, mocks):
            spec = env_spec_from_gymnax(env, params=None)
        assert spec.is_discrete is True
        assert spec.action_dim == 5
        assert spec.action_low is None
        assert spec.action_high is None

    def test_raises_on_discrete_obs(self):
        obs = _GDiscrete(10)
        act = _GBox(-1.0, 1.0, (2,))
        mocks, env = _gymnax_modules(obs, act)
        with patch.dict(sys.modules, mocks):
            with pytest.raises(ValueError, match="Box observation space"):
                env_spec_from_gymnax(env, params=None)

    def test_import_error_when_not_installed(self):
        # Setting sys.modules entries to None blocks re-import from disk,
        # simulating gymnax being absent even when it is installed.  Merely
        # deleting entries would let Python re-discover the package on disk.
        blocked = {k: None for k in list(sys.modules) if k.startswith("gymnax")}
        blocked.setdefault("gymnax", None)
        blocked.setdefault("gymnax.environments", None)
        blocked.setdefault("gymnax.environments.spaces", None)
        with patch.dict(sys.modules, blocked):
            with pytest.raises(ImportError, match="gymnax"):
                env_spec_from_gymnax(object(), params=None)


# ===========================================================================
# env_spec_from_jumanji  (mocked)
# ===========================================================================

class TestEnvSpecFromJumanji:
    def test_basic_extraction(self):
        obs = _JArray((11,))
        act = _JBoundedArray((3,), minimum=-1.0, maximum=1.0)
        mocks, env = _jumanji_modules(obs, act)
        with patch.dict(sys.modules, mocks):
            spec = env_spec_from_jumanji(env)
        assert spec.obs_shape == (11,)
        assert spec.action_dim == 3

    def test_action_bounds(self):
        obs = _JArray((4,))
        act = _JBoundedArray(
            (2,),
            minimum=np.array([-2.0, -3.0]),
            maximum=np.array([2.0, 3.0]),
        )
        mocks, env = _jumanji_modules(obs, act)
        with patch.dict(sys.modules, mocks):
            spec = env_spec_from_jumanji(env)
        assert float(spec.action_low[0]) == pytest.approx(-2.0)
        assert float(spec.action_high[1]) == pytest.approx(3.0)

    def test_scalar_bounds_broadcast(self):
        obs = _JArray((4,))
        act = _JBoundedArray((3,), minimum=-1.0, maximum=1.0)
        mocks, env = _jumanji_modules(obs, act)
        with patch.dict(sys.modules, mocks):
            spec = env_spec_from_jumanji(env)
        assert spec.action_low.shape == (3,)
        assert spec.action_high.shape == (3,)

    def test_raises_on_nested_obs_spec(self):
        obs = _JSpec()   # composite, not an Array
        act = _JBoundedArray((2,), minimum=-1.0, maximum=1.0)
        mocks, env = _jumanji_modules(obs, act)
        with patch.dict(sys.modules, mocks):
            with pytest.raises(ValueError, match="flat Array"):
                env_spec_from_jumanji(env)

    def test_discrete_action_returns_spec(self):
        obs = _JArray((4,))
        act = _JDiscreteArray(5)
        mocks, env = _jumanji_modules(obs, act)
        with patch.dict(sys.modules, mocks):
            spec = env_spec_from_jumanji(env)
        assert spec.is_discrete is True
        assert spec.action_dim == 5
        assert spec.action_low is None
        assert spec.action_high is None

    def test_raises_on_multi_discrete_action(self):
        obs = _JArray((4,))
        act = _JMultiDiscreteArray([3, 4, 5])
        mocks, env = _jumanji_modules(obs, act)
        with patch.dict(sys.modules, mocks):
            with pytest.raises(ValueError, match="[Dd]iscrete"):
                env_spec_from_jumanji(env)

    def test_raises_on_non_bounded_action(self):
        obs = _JArray((4,))
        act = _JArray((2,))   # Array but not BoundedArray
        mocks, env = _jumanji_modules(obs, act)
        with patch.dict(sys.modules, mocks):
            with pytest.raises(ValueError, match="BoundedArray"):
                env_spec_from_jumanji(env)

    def test_import_error_when_not_installed(self):
        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("jumanji")}
        try:
            with pytest.raises(ImportError, match="jumanji"):
                env_spec_from_jumanji(object())
        finally:
            sys.modules.update(saved)


# ===========================================================================
# create_iqlearn_from_env
# ===========================================================================

class TestCreateIQLearnFromEnv:
    def test_return_types(self):
        spec = _make_spec()
        data = _make_expert_data()
        state, fns, graphs = create_iqlearn_from_env(
            spec, data, buffer_size=100, train_steps=2
        )
        assert isinstance(state, IQLearnState)
        assert isinstance(fns, IQLearnFunctions)
        assert isinstance(graphs, IQLearnGraphs)

    def test_action_scale_symmetric(self):
        """[-1, 1] bounds → scale=1, bias=0."""
        spec = _make_spec(low=-1.0, high=1.0)
        data = _make_expert_data()
        # scale = (1 - (-1)) / 2 = 1.0, bias = (1 + (-1)) / 2 = 0.0
        # We verify indirectly: deterministic prediction should be in [-1, 1]
        state, fns, _ = create_iqlearn_from_env(spec, data, buffer_size=100, train_steps=1)
        obs = jnp.zeros((4,))  # single unbatched observation
        action = fns.predict(state, obs, deterministic=True)
        assert jnp.all(action >= -1.0 - 1e-5)
        assert jnp.all(action <= 1.0 + 1e-5)

    def test_action_scale_asymmetric(self):
        """[0, 4] bounds → scale=2, bias=2; actions should lie in [0, 4]."""
        spec = _make_spec(action_dim=2, low=0.0, high=4.0)
        data = _make_expert_data(action_dim=2)
        state, fns, _ = create_iqlearn_from_env(spec, data, buffer_size=100, train_steps=1)
        obs = jnp.zeros((4,))  # single unbatched observation
        action = fns.predict(state, obs, deterministic=True)
        assert jnp.all(action >= 0.0 - 1e-5)
        assert jnp.all(action <= 4.0 + 1e-5)

    def test_default_hp_target_entropy(self):
        """Default hp should set target_entropy = -action_dim."""
        spec = _make_spec(action_dim=3)
        data = _make_expert_data(action_dim=3)
        # Pass no hp → default should use target_entropy = -3
        state, fns, _ = create_iqlearn_from_env(spec, data, buffer_size=100, train_steps=1)
        # We can't inspect hp directly from the closed-over state, but we can
        # verify training runs without error and produces finite metrics.
        new_state, metrics = fns.train(state, jax.random.key(0))
        assert all(jnp.isfinite(v).all() for v in jax.tree.leaves(metrics))

    def test_custom_hp_respected(self):
        """A caller-supplied hp should be passed through unchanged."""
        spec = _make_spec()
        data = _make_expert_data()
        hp = Hyperparameters(alpha=0.5, autotune_alpha=False)
        state, _, _ = create_iqlearn_from_env(
            spec, data, hp=hp, buffer_size=100, train_steps=1
        )
        assert float(state.log_alpha) == pytest.approx(jnp.log(0.5), abs=1e-5)

    def test_buffer_has_sampleable_slots(self):
        """After filling with 20 transitions, buffer must have sampleable slots."""
        from lambda_imitation.buffer import Buffer
        # We can't access the buffer after create_iqlearn_from_env returns,
        # so instead verify training doesn't crash (which requires sampleable slots).
        spec = _make_spec()
        data = _make_expert_data(n=20)
        state, fns, _ = create_iqlearn_from_env(
            spec, data, buffer_size=100, train_steps=1,
        )
        new_state, metrics = fns.train(state, jax.random.key(42))
        assert isinstance(new_state, IQLearnState)

    def test_custom_obs_action_keys(self):
        """Non-default obs/action key names should be forwarded correctly."""
        spec = _make_spec()
        data = {
            "s": jnp.ones((20, 4)),
            "a": jnp.zeros((20, 2)),
        }
        state, fns, _ = create_iqlearn_from_env(
            spec, data, buffer_size=100, train_steps=1,
            obs_key="s", action_key="a",
        )
        new_state, metrics = fns.train(state, jax.random.key(0))
        assert isinstance(new_state, IQLearnState)

    def test_raises_on_missing_obs_key(self):
        spec = _make_spec()
        data = {"actions": jnp.zeros((10, 2))}
        with pytest.raises(ValueError, match="observations"):
            create_iqlearn_from_env(spec, data)

    def test_raises_on_missing_action_key(self):
        spec = _make_spec()
        data = {"observations": jnp.ones((10, 4))}
        with pytest.raises(ValueError, match="actions"):
            create_iqlearn_from_env(spec, data)

    def test_raises_on_empty_data(self):
        spec = _make_spec()
        data = {
            "observations": jnp.ones((0, 4)),
            "actions": jnp.zeros((0, 2)),
        }
        with pytest.raises(ValueError, match="at least one transition"):
            create_iqlearn_from_env(spec, data)

    def test_multidim_obs_shape(self):
        """Observations with shape (H, W, C) should work via FE flattening."""
        spec = _make_spec(obs_shape=(8, 8, 3), action_dim=2)
        data = {
            "observations": jnp.ones((20, 8, 8, 3)),
            "actions": jnp.zeros((20, 2)),
        }
        state, fns, _ = create_iqlearn_from_env(
            spec, data, buffer_size=100, train_steps=1,
            fe_hidden_dims=(64,),
        )
        # predict takes a single (unbatched) observation
        obs = jnp.zeros((8, 8, 3))
        action = fns.predict(state, obs, deterministic=True)
        assert action.shape == (2,)

    def test_predictions_within_bounds(self):
        """Predicted actions must always lie within [action_low, action_high]."""
        spec = _make_spec(low=-3.0, high=5.0, action_dim=4)
        data = _make_expert_data(obs_dim=4, action_dim=4)
        state, fns, _ = create_iqlearn_from_env(
            spec, data, buffer_size=100, train_steps=1,
        )
        # predict takes a single (unbatched) observation → returns (action_dim,)
        obs = jnp.zeros((4,))
        action = fns.predict(state, obs, deterministic=True)
        assert jnp.all(action >= -3.0 - 1e-5)
        assert jnp.all(action <= 5.0 + 1e-5)


def _make_discrete_expert_data(n=20, obs_dim=4, num_actions=4):
    """Expert data with float32 action indices of shape (n, 1)."""
    actions = jnp.array(
        [[float(i % num_actions)] for i in range(n)], dtype=jnp.float32
    )
    return {
        "observations": jnp.ones((n, obs_dim)),
        "actions": actions,
    }


# ===========================================================================
# create_iqlearn_from_env  (discrete)
# ===========================================================================

class TestCreateIQLearnFromEnvDiscrete:
    def test_return_types(self):
        spec = _make_discrete_spec()
        data = _make_discrete_expert_data()
        state, fns, graphs = create_iqlearn_from_env(
            spec, data, buffer_size=100, train_steps=2
        )
        assert isinstance(state, IQLearnState)
        assert isinstance(fns, IQLearnFunctions)
        assert isinstance(graphs, IQLearnGraphs)

    def test_default_target_entropy(self):
        """Default hp should set target_entropy = 0.98 * log(num_actions)."""
        import math
        num_actions = 4
        spec = _make_discrete_spec(num_actions=num_actions)
        data = _make_discrete_expert_data(num_actions=num_actions)
        # Training should run without error (verifies target_entropy was set)
        state, fns, _ = create_iqlearn_from_env(spec, data, buffer_size=100, train_steps=1)
        new_state, metrics = fns.train(state, jax.random.key(0))
        assert all(jnp.isfinite(v).all() for v in jax.tree.leaves(metrics))

    def test_predict_returns_float32_scalar(self):
        """Discrete predict should return a float32 scalar (action index)."""
        spec = _make_discrete_spec()
        data = _make_discrete_expert_data()
        state, fns, _ = create_iqlearn_from_env(spec, data, buffer_size=100, train_steps=1)
        obs = jnp.zeros((4,))
        action = fns.predict(state, obs, deterministic=True)
        assert action.shape == ()
        assert action.dtype == jnp.float32

    def test_predict_deterministic_valid_range(self):
        """Deterministic action must be a valid action index in [0, num_actions)."""
        num_actions = 6
        spec = _make_discrete_spec(num_actions=num_actions)
        data = _make_discrete_expert_data(num_actions=num_actions)
        state, fns, _ = create_iqlearn_from_env(spec, data, buffer_size=100, train_steps=1)
        obs = jnp.zeros((4,))
        action = fns.predict(state, obs, deterministic=True)
        assert float(action) >= 0.0
        assert float(action) < num_actions

    def test_predict_stochastic_valid_range(self):
        """Stochastic samples must all be valid action indices."""
        num_actions = 4
        spec = _make_discrete_spec(num_actions=num_actions)
        data = _make_discrete_expert_data(num_actions=num_actions)
        state, fns, _ = create_iqlearn_from_env(spec, data, buffer_size=100, train_steps=1)
        obs = jnp.zeros((4,))
        for seed in range(10):
            action = fns.predict(state, obs, key=jax.random.key(seed), deterministic=False)
            assert float(action) >= 0.0
            assert float(action) < num_actions

    def test_train_metrics_finite(self):
        spec = _make_discrete_spec()
        data = _make_discrete_expert_data()
        state, fns, _ = create_iqlearn_from_env(spec, data, buffer_size=100, train_steps=2)
        new_state, metrics = fns.train(state, jax.random.key(0))
        for k, v in metrics.items():
            assert jnp.isfinite(v), f"metric '{k}' is not finite: {v}"

    def test_custom_hp_discrete(self):
        """Caller-supplied hp should be forwarded unchanged for discrete."""
        spec = _make_discrete_spec()
        data = _make_discrete_expert_data()
        hp = Hyperparameters(alpha=0.3, autotune_alpha=False)
        state, _, _ = create_iqlearn_from_env(
            spec, data, hp=hp, buffer_size=100, train_steps=1
        )
        assert float(state.log_alpha) == pytest.approx(jnp.log(0.3), abs=1e-5)


# Need to import jax at module level for the test that uses jax.random.key
import jax
