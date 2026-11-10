"""Independent explicit-window oracle, information caps and learning routing."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from lambda_imitation.history import HistoryFeatureExtractor
from lambda_imitation import EnvSpec,Hyperparameters,create_actor_critic_from_env

HISTORIES=(0,1,2,5,10,15,20,25)


def model(h):
    return HistoryFeatureExtractor((1,),4,4,2,h,rngs=nnx.Rngs(17))


@pytest.mark.parametrize('h',HISTORIES)
def test_vectorized_matches_independent_from_zero_window_and_online(h):
    m=model(h);T=36
    obs=jnp.arange(T,dtype=jnp.float32).reshape(T,1,1)*.1
    acts=jax.nn.one_hot(jnp.arange(T).reshape(T,1)%2,2)
    dones=jnp.zeros((T,1)).at[8,0].set(1).at[23,0].set(1)
    actual=m.sequence(obs,acts,dones)
    history=m.initialize_carry(1);pa=jnp.zeros((1,2));episode_start=0
    online=[];oracle=[]
    for t in range(T):
        history,y=m(history,obs[t],pa);online.append(y)
        carry=jnp.zeros((1,4))
        for k in range(max(episode_start,t-h),t+1):
            prev=jnp.zeros((1,2)) if k==episode_start else acts[k-1]
            carry,out=m.core(carry,obs[k],prev)
        oracle.append(out)
        if bool(dones[t,0]):
            history=jnp.zeros_like(history);pa=jnp.zeros_like(pa);episode_start=t+1
        else:pa=acts[t]
    np.testing.assert_allclose(actual,jnp.stack(oracle),atol=2e-6,rtol=2e-5)
    np.testing.assert_allclose(actual,jnp.stack(online),atol=2e-6,rtol=2e-5)
    np.testing.assert_allclose(m.sequence(obs,acts,dones,start=15),actual[15:],atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('h',HISTORIES)
def test_no_older_information_no_future_leak(h):
    m=model(h);T=36
    obs=jnp.arange(T,dtype=jnp.float32).reshape(T,1,1)*.1
    acts=jax.nn.one_hot(jnp.arange(T).reshape(T,1)%2,2);d=jnp.zeros((T,1))
    original=m.sequence(obs,acts,d)
    # Last network token includes obs and PREVIOUS action. Only change raw data
    # strictly older than the earliest token's previous action.
    changed=m.sequence(obs.at[:T-1-h].set(-50),acts.at[:max(0,T-2-h)].set(0),d)
    np.testing.assert_array_equal(original[-1],changed[-1])
    changed=m.sequence(obs.at[20:].set(-50),acts,d)
    np.testing.assert_array_equal(original[:20],changed[:20])


def test_identical_parameter_count_and_initialization_every_h():
    states=[nnx.split(model(h))[1] for h in HISTORIES]
    for s in states[1:]:
        for a,b in zip(jax.tree.leaves(states[0]),jax.tree.leaves(s)):
            np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('mode',('greedy','sampled'))
def test_heldout_rms_oracle_and_diagnostic_does_not_change_policy_rng(mode):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
    from _common import Experiment,make_evaluator
    from lambda_imitation.envs.battleship import Battleship
    env=Battleship()
    hp=Hyperparameters(batch_size=2,online_buffer_size=160,history_context=26)
    state,fns=create_actor_critic_from_env(EnvSpec((26,),25),hp=hp,projection=4,
        memory_type='gru',memory_hidden_dim=4,use_prev_action=True,actor_dims=(),critic_dims=(),
        lambda1_critic_dims=(),lambda2_critic_dims=(),history_length=1,
        obs_fn=lambda o:o[...,:1],mask_fn=lambda o:o[...,1:])
    # Independent constant-head oracle: both lambda1 twins = 0, lambda2 = 2.
    zero=jax.tree.map(jnp.zeros_like,state.lambda1_critic)
    two=jax.tree.map(lambda x:jnp.full_like(x,2) if x.ndim==1 else jnp.zeros_like(x),state.lambda2_critic)
    state=state._replace(lambda1_critic=zero,lambda2_critic=two)
    experiment=Experiment(env,4,(),(),26,obs_fn=lambda o:o[...,:1],mask_fn=lambda o:o[...,1:])
    keys=jnp.stack([jax.random.PRNGKey(19)])
    batched=jax.tree.map(lambda x:jnp.asarray(x)[None],state)
    got=make_evaluator(experiment,fns,27,25,3,mode)(batched,keys)
    without=make_evaluator(experiment,fns._replace(discrepancy=None),27,25,3,mode)(batched,keys)
    np.testing.assert_allclose(got['discrepancy_rms'],2,atol=1e-6)
    np.testing.assert_array_equal(got['aux_twin_rms'],0)
    for metric in ('return','steps','completed','repeat_fraction'):
        np.testing.assert_array_equal(got[metric],without[metric])


@pytest.mark.parametrize('h',(0,2,25))
def test_learner_gradient_routing_and_loss_lengths(h):
    hp=Hyperparameters(batch_size=2,online_buffer_size=160,burn_in_length=10,
        sequence_length=40,lambda_truncation=50,history_context=26,lambda_critic_loss='mse')
    state,fns,debug=create_actor_critic_from_env(EnvSpec((1,),2),hp=hp,
        projection=4,memory_type='gru',memory_hidden_dim=4,use_prev_action=True,
        actor_dims=(4,),critic_dims=(4,),lambda1_critic_dims=(4,),lambda2_critic_dims=(4,),
        history_length=h,train_steps=2,seed=7,debug=True)
    info={**state.online_buffer.info,'observations':jnp.arange(160,dtype=jnp.float32)[:,None]*.01,
        'actions':(jnp.arange(160)%2).astype(jnp.float32)[:,None],
        'rewards':jnp.ones(160),'behaviour_weight':jnp.full(160,.5),
        'terminated':(jnp.arange(160)%24==0).astype(jnp.float32)}
    state=state._replace(online_buffer=state.online_buffer._replace(info=info,sampling_ok=jnp.ones(160,bool),pos=160))
    key=jax.random.key(9)
    grad=jax.grad(lambda fe:debug.loss(fe,*state[1:5],state,key)[1]['actor_loss'])(state.feature_extractor)
    assert all(np.all(np.asarray(x)==0) for x in jax.tree.leaves(grad))
    grad=jax.grad(lambda fe:debug.loss(fe,*state[1:5],state,key)[1]['ld_loss'])(state.feature_extractor)
    assert any(np.any(np.asarray(x)!=0) for x in jax.tree.leaves(grad))
    obs=info['observations'][:126][None].repeat(2,0)
    actions=info['actions'][:126][None].repeat(2,0)
    dones=info['terminated'][:126][None].repeat(2,0)
    latent,target=debug.calculate_latent(state.feature_extractor,state.feature_extractor_target,
        obs,actions,dones,jnp.zeros((2,h*4)))
    assert latent.shape==target.shape==(90,2,4)
    after,metrics=debug.update_step(state,key)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((after,metrics)))
    assert int(after.update_step)==1
    d2,t2=fns.discrepancy(state,jnp.ones(1),jnp.zeros(h*4),jnp.zeros(2))
    assert np.isfinite([d2,t2]).all() and d2>=0 and t2>=0
