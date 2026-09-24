"""The PocMan --paper-arch network must match the reference implementation.

Reference: ``DiscreteActorCriticRNN`` in ``lamb/models.py`` of
brownirl/lambda_discrepancy, with PocMan's selected hyperparameters
(``pocman_LD_ppo_best.py``: ``hidden_size=512, action_concat=True,
double_critic=True``):

    embedding  Dense(H) -> ReLU      over concat([obs, prev_action_onehot])
    memory     GRUCell(H)
    actor      Dense(H) -> ReLU -> Dense(num_actions)
    critic     Dense(H) -> ReLU -> Dense(1), two independent copies
    and no LayerNorm anywhere

This pins the wiring, which is the part that silently drifts.
"""

import jax
import jax.numpy as jnp
import pytest

from lambda_imitation.iqlearn import Hyperparameters
from lambda_imitation.utils import (create_iqlearn_from_env, EnvSpec,
                                    relu_projection)

H = 512
OBS_DIM, NUM_ACTIONS = 11, 4          # PocMan's observation and action space

def _kernels(tree):
    return {"".join(str(k) for k in path): tuple(leaf.shape)
            for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
            if getattr(leaf, "ndim", 0) == 2}

@pytest.fixture(scope="module")
def paper_state():
    spec = EnvSpec(obs_shape=(OBS_DIM,), action_dim=NUM_ACTIONS,
                   action_low=None, action_high=None, is_discrete=True)
    expert = {"observations": jnp.zeros((1, OBS_DIM), dtype=jnp.float32),
              "actions": jnp.zeros((1, 1), dtype=jnp.float32)}
    state, _, _ = create_iqlearn_from_env(
        spec, expert, buffer_size=1,
        hp=Hyperparameters(lambda1=0.5, lambda2=0.95, gamma=0.95),
        projection=relu_projection(H), memory_type="gru", memory_hidden_dim=H,
        actor_dims=(H,), critic_dims=(H,), lambda1_critic_dims=(H,),
        lambda2_critic_dims=(H,), train_steps=2, approximate_lambda=True,
        use_prev_action=True, critic_layer_norm=False, debug=True, seed=0,
        use_sac=False)
    return state

def test_embedding_takes_obs_and_prev_action(paper_state):
    """Dense(H) over [obs | prev-action one-hot] — the paper's action_concat."""
    fe = _kernels(paper_state.feature_extractor)
    projection = [v for k, v in fe.items() if "projection" in k]
    assert projection == [(OBS_DIM + NUM_ACTIONS, H)]

def test_memory_is_a_gru_of_width_h(paper_state):
    fe = _kernels(paper_state.feature_extractor)
    cell = sorted(v for k, v in fe.items() if "cell" in k)
    assert cell == [(H, 3 * H), (H, 3 * H)]      # input and hidden, 3 gates

def test_heads_have_one_hidden_layer_of_width_h(paper_state):
    assert list(_kernels(paper_state.actor).values()) == [(H, H), (H, NUM_ACTIONS)]
    critic = _kernels(paper_state.critic)
    # twin critic: both branches are H -> H -> per-action values
    assert sorted(critic.values()) == [(H, NUM_ACTIONS), (H, NUM_ACTIONS), (H, H), (H, H)]

def test_no_layer_norm_anywhere(paper_state):
    names = ["".join(str(k) for k in p) for p, _ in
             jax.tree_util.tree_flatten_with_path(paper_state.critic)[0]]
    assert not [n for n in names if "norm" in n.lower()]

def test_relu_projection_actually_applies_the_relu():
    """Our default LinearProjection has no activation; the paper's does."""
    from flax import nnx
    from lambda_imitation.utils import LinearProjection, ReluProjection

    rngs = nnx.Rngs(0)
    obs = jnp.full((2, OBS_DIM), -5.0)           # drive the pre-activation negative
    prev = jnp.zeros((2, NUM_ACTIONS))
    plain = LinearProjection(OBS_DIM, NUM_ACTIONS, H, rngs=nnx.Rngs(0))
    relu = ReluProjection(H, OBS_DIM, NUM_ACTIONS, rngs=nnx.Rngs(0))
    assert (relu(obs, prev) >= 0).all()
    assert (plain(obs, prev) < 0).any()
