---
name: add-env-adapter
description: Add a new env_spec_from_* extractor for a new RL library (e.g. gymnasium, gymnax, jumanji). Covers utils.py implementation, __init__.py re-export, and test_utils.py test class.
---

## What this skill covers

Adding support for a new RL library means touching three files in a fixed pattern:

1. `src/lambda_imitation/utils.py` — implement `env_spec_from_<lib>()`
2. `src/lambda_imitation/__init__.py` — re-export the new function
3. `tests/test_utils.py` — add a `TestEnvSpecFrom<Lib>` test class

Run `pytest tests/test_utils.py` to verify when done.

---

## Step 1 — implement the extractor in utils.py

Place the new function after the existing extractors, before the
`create_iqlearn_from_env` factory. Follow this template exactly:

```python
def env_spec_from_<lib>(env, ...) -> EnvSpec:
    """Extract an EnvSpec from a <lib> environment.

    <one-paragraph description of what is read and how>

    Args:
        env: ...
        ...

    Returns:
        An EnvSpec populated from the environment's spaces/specs.
        is_discrete=True and action_low=action_high=None for discrete spaces.

    Raises:
        ImportError: If <lib> is not installed.
        ValueError: If the observation space is not supported, or the action
            space is neither continuous (Box/BoundedArray) nor discrete.
    """
    try:
        import <lib>.<spaces_or_specs_module> as spaces
    except ImportError as exc:
        raise ImportError(
            "<lib> is required for env_spec_from_<lib>. "
            "Install it with: pip install <lib>"
        ) from exc

    obs_space = ...   # extract observation space/spec from env
    act_space = ...   # extract action space/spec from env

    if not isinstance(obs_space, spaces.<BoxEquivalent>):
        raise ValueError(
            f"env_spec_from_<lib> requires a <Box-equivalent> observation space, "
            f"got {type(obs_space).__name__}."
        )

    # Discrete branch — check BEFORE BoundedArray (DiscreteArray is a subclass)
    if isinstance(act_space, spaces.<DiscreteEquivalent>):
        return EnvSpec(
            obs_shape=tuple(obs_space.shape),
            action_dim=int(act_space.<n_or_num_values>),
            is_discrete=True,
        )

    # Continuous branch
    if isinstance(act_space, spaces.<BoundedEquivalent>):
        action_dim = math.prod(act_space.shape)
        low  = jnp.broadcast_to(jnp.asarray(act_space.<low>,  dtype=jnp.float32), (action_dim,))
        high = jnp.broadcast_to(jnp.asarray(act_space.<high>, dtype=jnp.float32), (action_dim,))
        return EnvSpec(
            obs_shape=tuple(obs_space.shape),
            action_dim=action_dim,
            is_discrete=False,
            action_low=low,
            action_high=high,
        )

    raise ValueError(
        f"env_spec_from_<lib> requires a <Bounded> or <Discrete> action space, "
        f"got {type(act_space).__name__}."
    )
```

Key rules:
- Import lazily inside the function body (not at module top level).
- Always check DiscreteArray **before** BoundedArray — in some libraries
  DiscreteArray is a subclass of BoundedArray (e.g. jumanji).
- Use `jnp.broadcast_to` for scalar bounds so arrays are always shape `(action_dim,)`.
- `action_low` / `action_high` must be `None` for discrete spaces (omit them).
- All returned arrays must be `dtype=jnp.float32`.

---

## Step 2 — re-export from __init__.py

Open `src/lambda_imitation/__init__.py`. Add the new function name to the
`from .utils import ...` line (keep names alphabetically sorted within each
import group).

---

## Step 3 — write the test class in test_utils.py

Add a new class `TestEnvSpecFrom<Lib>` after the existing `TestEnvSpecFrom*`
classes and before `TestCreateIQLearnFromEnv`.

Mandatory test methods (mirror the existing classes):

| Method | What it checks |
|---|---|
| `test_basic_extraction` | `obs_shape`, `action_dim`, `is_discrete=False` for a continuous env |
| `test_action_bounds` | `action_low` and `action_high` values are correct |
| `test_scalar_bounds_broadcast` | scalar bounds broadcast to shape `(action_dim,)` |
| `test_discrete_action_returns_spec` | discrete env → `is_discrete=True`, `action_low/high` are `None` |
| `test_raises_on_discrete_obs` | non-Box obs space raises `ValueError` |
| `test_import_error_when_not_installed` | `sys.modules` patch makes the import fail → `ImportError` |

Mocking pattern (required — do NOT use `MagicMock` for parent modules):

```python
import sys
import types

@pytest.fixture
def mock_<lib>(monkeypatch):
    # Build a minimal module tree so dotted imports resolve correctly.
    parent = types.ModuleType("<lib>")
    spaces_mod = types.ModuleType("<lib>.<spaces_module>")

    class FakeBox:
        def __init__(self, shape, low, high):
            self.shape = shape
            self.low = low
            self.high = high

    class FakeDiscrete:
        def __init__(self, n):
            self.n = n        # or .num_values depending on the library
            self.shape = ()

    spaces_mod.Box = FakeBox
    spaces_mod.Discrete = FakeDiscrete
    parent.<spaces_attr> = spaces_mod   # e.g. parent.spaces = spaces_mod

    monkeypatch.setitem(sys.modules, "<lib>", parent)
    monkeypatch.setitem(sys.modules, "<lib>.<spaces_module>", spaces_mod)
    return spaces_mod
```

Then each test receives `mock_<lib>` as a fixture and constructs fake env
objects using the fake space classes.

The `test_import_error_when_not_installed` test removes the module from
`sys.modules` to simulate it being absent:

```python
def test_import_error_when_not_installed(self, monkeypatch):
    monkeypatch.delitem(sys.modules, "<lib>", raising=False)
    monkeypatch.delitem(sys.modules, "<lib>.<spaces_module>", raising=False)

    class FakeEnv:
        pass

    with pytest.raises(ImportError, match="<lib>"):
        env_spec_from_<lib>(FakeEnv())
```

---

## Checklist

- [ ] `env_spec_from_<lib>` implemented in `utils.py` with lazy import
- [ ] Discrete branch checked before continuous (subclass safety)
- [ ] Bounds broadcast to `(action_dim,)` float32
- [ ] `action_low` / `action_high` are `None` for discrete
- [ ] Function added to `__init__.py` re-exports
- [ ] `TestEnvSpecFrom<Lib>` class added to `test_utils.py` with all 6 methods
- [ ] `pytest tests/test_utils.py` passes (131 → 131 + N tests)
- [ ] `pytest tests/` still passes in full
