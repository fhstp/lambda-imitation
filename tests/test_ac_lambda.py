"""Small CPU proofs; optional live-study oracle is read-only and hash-bound."""

import ast
import importlib
import json
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lambda_imitation import ac_lambda as ac, ac_lambda_train as run


def same(a, b, *, exact=False):
    left, right = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(left) == len(right)
    for x, y in zip(left, right, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape and x.dtype == y.dtype
        assert np.isfinite(x).all() and np.isfinite(y).all()
        if exact or x.dtype.kind not in "fc":
            np.testing.assert_array_equal(x, y)
        else:
            np.testing.assert_allclose(x, y, atol=1e-6, rtol=1e-6)


@pytest.fixture(scope="module")
def fixture():
    model, learner = ac.initialize_model(9, width=8)
    env, state = ac.initial_state(model, learner, 9, capacity=128)
    prefill, _ = ac.kernels(model, env)
    state = jax.block_until_ready(prefill(state, 80))
    _, batch, _ = ac.replay.sample(state.replay, jax.random.PRNGKey(17), 2, 25)
    assert bool(state.healthy) and bool(batch.integrity)
    return model, env, state, batch


def test_config_and_cli():
    c = ac.resolved_config()
    assert c["settings"]["fe_learning_rate"] == 1e-5
    assert c["settings"]["head_learning_rate"] == 1e-4
    assert ac.resolved_config(total=2000000)["checkpoints"][-1] == 2000000
    for seed, total in ((-1,500000),(0,1000000),(True,500000)):
        with pytest.raises(ValueError):
            ac.resolved_config(seed,total)
    with pytest.raises(SystemExit):
        run.parser().parse_args(["--output","unused","--gvd"])
    assert set(ac.GROUPS) == {"actor_fe","actor","standard","lambda1","lambda2"}


def test_next_action_recursion_and_terminal(fixture):
    _, _, _, batch = fixture
    shape = (*batch.actions.shape,25,2)
    q = jax.random.normal(jax.random.PRNGKey(3), shape)
    pi, _ = ac.policy(jax.random.normal(jax.random.PRNGKey(4),shape[:-1]),batch.observations[...,1:])
    actual = ac.lambda_targets(batch,q,pi,ac.Settings())
    qn, pn = np.asarray(q), np.asarray(pi)
    rewards, valid, done, action, mu = map(np.asarray,(batch.rewards,batch.valid,batch.dones,batch.actions,batch.behavior))
    expected = np.zeros_like(actual)
    for b in range(action.shape[1]):
        for t in reversed(range(action.shape[0])):
            if not valid[t,b]: continue
            expected[t,b] = rewards[t,b]
            if not done[t,b]:
                n = t+1
                v = (pn[n,b,:,None]*qn[n,b]).sum(axis=0)
                c = np.array([.1,.95])*min(1.1,pn[n,b,action[n,b]]/mu[n,b])
                expected[t,b] += .99*(v+c*(expected[n,b]-qn[n,b,action[n,b]]))
    np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)
    assert not np.any(jax.grad(lambda x:ac.lambda_targets(batch,x,pi,ac.Settings()).sum())(q))
    np.testing.assert_array_equal(np.asarray(actual)[done & valid],25.)


def test_gradient_routes_and_mask(fixture):
    model, _, state, batch = fixture
    grads = jax.jit(jax.jacrev(lambda p:ac.objectives(model,p,state.learner.target,batch)[0]))(state.learner.online)
    norms = {name:np.array([sum(float(jnp.sum(x[i]**2)) for x in jax.tree.leaves(tree)) for i in range(5)])
             for name,tree in grads.items()}
    assert norms["actor_fe"][0] > 0 and norms["actor_fe"][4] > 0
    for name in ("standard","lambda1","lambda2"):
        assert norms[name][0] == 0 and norms[name][4] == 0
    assert norms["actor"][0] > 0 and np.all(norms["actor"][1:] == 0)
    assert norms["lambda1"][2] > 0 and norms["lambda2"][3] > 0
    pi, logpi = ac.policy(jnp.ones((2,25)),jnp.array([[True]+[False]*24,[False]*25]))
    assert pi[0,0] == 1 and jnp.isfinite(logpi).all()
    same(jax.grad(lambda p:ac.objectives(model,state.learner.online,p,batch)[0].sum())(state.learner.target),
         jax.tree.map(jnp.zeros_like,state.learner.target),exact=True)


def test_replay_eviction_and_rollover(fixture):
    model, env, _, _ = fixture
    _, learner = ac.initialize_model(9,width=8)
    _, state = ac.initial_state(model,learner,9,capacity=30)
    prefill,_ = ac.kernels(model,env)
    state = prefill(state,100)
    replay,batch,_ = ac.replay.sample(state.replay,jax.random.PRNGKey(2),128,25)
    assert bool(batch.integrity) and int(replay.evicted) > 0
    assert np.all(np.asarray(batch.insertion_ids)[np.asarray(batch.valid)] >= 70)
    assert int(replay.episode_draws) == 128
    assert int(replay.presentations[0]) == int(batch.valid.sum())
    np.testing.assert_array_equal(ac.replay.add_count(jnp.array([2**32-2,1],jnp.uint32),5),[3,2])


def test_checkpoint_exact_resume_and_rejections(fixture,tmp_path):
    model, env, state, _ = fixture
    _, train = ac.kernels(model,env)
    first,_ = train(state,1)
    config = ac.resolved_config(9)
    path = tmp_path/"full"
    run.save_checkpoint(path,first,config,full=True)
    restored = run.restore_checkpoint(path,state,ac.resolved_config(9,2000000))
    same(first,restored,exact=True)
    same(train(first,1),train(restored,1),exact=True)
    with pytest.raises(FileExistsError): run.save_checkpoint(path,first,config,full=True)
    with pytest.raises(ValueError): run.restore_checkpoint(path,state,ac.resolved_config(0))
    run.save_checkpoint(tmp_path/"compact",first,config,full=False)
    with pytest.raises(ValueError): run.restore_checkpoint(tmp_path/"compact",state,config)
    (path/"state.msgpack").write_bytes(b"corrupt")
    with pytest.raises(ValueError): run.restore_checkpoint(path,state,config)


def test_evaluation_isolation(fixture):
    model, env, state, _ = fixture
    before = [np.asarray(x).copy() for x in jax.tree.leaves(state)]
    rows = jax.jit(lambda p:ac.evaluate(model,env,p,*ac.evaluation_keys(9,3)))(state.learner.online)
    assert np.asarray(rows["completed"]).all() and np.asarray(rows["legal"]).all()
    np.testing.assert_array_equal(rows["returns"],26-rows["lengths"])
    same(before,jax.tree.leaves(state),exact=True)


def test_entrypoint_fresh_and_resume_schedule(fixture,tmp_path,monkeypatch):
    """Real CLI/writer/loader; neural kernels mocked only for long-horizon routing."""
    model,env,_,_ = fixture
    _,learner = ac.initialize_model(9,width=8)
    _,initial = ac.initial_state(model,learner,9,capacity=128)
    prefill_calls = []
    def prefill(state,n):
        prefill_calls.append(n)
        return state._replace(prefill=jnp.int32(n),replay=state.replay._replace(inserted=jnp.int32(n)))
    def train(state,n):
        count = state.interactions+n
        opts = {k:(v[0]._replace(count=count),*v[1:]) for k,v in state.learner.optimizers.items()}
        return state._replace(interactions=count,replay=state.replay._replace(inserted=count+state.prefill),
            learner=state.learner._replace(updates=count,optimizers=opts)),{"loss":jnp.float32(0)}
    monkeypatch.setattr(ac,"initialize_model",lambda seed:(model,learner))
    monkeypatch.setattr(ac,"initial_state",lambda *args:(env,initial))
    monkeypatch.setattr(ac,"kernels",lambda *args:(prefill,train))
    monkeypatch.setattr(ac,"evaluate",lambda *args:dict(returns=jnp.full(500,6),lengths=jnp.full(500,20),
                                                      completed=jnp.ones(500,bool),legal=jnp.ones(500,bool)))
    first,second = tmp_path/"first",tmp_path/"second"
    monkeypatch.setattr(sys,"argv",["ac_lambda_train","--output",str(first),"--seed","9"])
    run.main()
    assert prefill_calls == [16640]
    assert sorted(p.name for p in first.iterdir() if p.is_dir()) == [f"step-{n:08d}" for n in (0,100000,250000,500000)]
    monkeypatch.setattr(sys,"argv",["ac_lambda_train","--output",str(second),"--seed","9","--total","2000000",
                                   "--resume",str(first/"step-00500000")])
    run.main()
    assert prefill_calls == [16640]
    assert sorted(p.name for p in second.iterdir() if p.is_dir()) == [f"step-{n:08d}" for n in (750000,1000000,1500000,2000000)]
    restored = run.restore_checkpoint(second/"step-02000000",initial,ac.resolved_config(9,2000000))
    assert int(restored.interactions) == 2000000 and int(restored.prefill) == 16640
    assert int(restored.evaluation_interactions) == 8*500*20


def test_active_study_parity(fixture):
    """Run the actual frozen F5 functions, not a copied oracle implementation."""
    study = os.environ.get("AC_LAMBDA_STUDY")
    if not study:
        pytest.skip("set AC_LAMBDA_STUDY to the approved immutable study snapshot")
    study = Path(study)
    # Exact active, non-abandoned computation source identities.
    hashes = {
        "battleship_fe_addons.py":"1b65723749b19b7d5f093efe507a1a5a76dd7f2a53de222f30b687270d453aa1",
        "battleship_explore.py":"de125e4a77427424db4496099321ad3f60b9603933267c1a809a509c10f02299",
        "battleship_combined.py":"40cf75d303b498d34c82436476e975f30f8dc1c4e514694dcf4823510e494c06",
    }
    for name,sha in hashes.items():
        assert run.digest((study/"experiments/battleship_ladder"/name).read_bytes()) == sha
    sys.path.append(str(study))
    f = importlib.import_module("experiments.battleship_ladder.battleship_fe_addons")
    # Shared canonical model classes must also match the active study definitions.
    here = Path(ac.__file__).parent
    for file,names in (("iqlearn.py",("Head","TwinCriticState")),
                       ("utils.py",("RecurrentFeatureExtractor","BattleshipProjection","battleship_projection"))):
        def definitions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                    if getattr(n,"name",None) in names}
        assert definitions(here/file) == definitions(study/"src/lambda_imitation"/file)
    current = json.loads((study/"experiments/battleship_ladder/fixtures/Battleship5x5ACLambdaContinuation-v1.json").read_bytes())
    config = ac.resolved_config(total=2000000)
    for name,value in json.loads(run.canonical(config["settings"])).items():
        if name in current["settings"]:
            assert value == current["settings"][name],name
    assert config["settings"]["fe_learning_rate"] == current["fe_learning_rate"]
    assert config["settings"]["head_learning_rate"] == current["head_learning_rate"]
    model,env,state,batch = fixture
    cm,core = f.e.five.initialize_model(9,width=8)
    old_model,old_learner = f.adapt(cm,core,f.Config("F5-lambda-off"))
    _,learner = ac.initialize_model(9,width=8)
    same(old_learner,learner,exact=True)
    same(model.unroll(learner.online["actor_fe"],batch),old_model.unroll(old_learner.online["actor_fe"],batch),exact=True)
    old_targets,_ = f.e.qc.control_targets(old_model,old_learner.target,batch,"QC-qtarget")
    same(ac.targets(model,learner.target,batch)[0],old_targets)
    same(ac.objectives(model,learner.online,learner.target,batch),f.objectives(old_model,old_learner.online,old_learner.target,batch))
    new_grad = jax.jit(jax.grad(lambda p:ac.objectives(model,p,learner.target,batch)[0].sum()))(learner.online)
    old_grad = jax.jit(jax.grad(lambda p:f.objectives(old_model,p,old_learner.target,batch)[0].sum()))(old_learner.online)
    same(new_grad,old_grad)
    new_update = jax.jit(lambda s:ac.update(model,s,batch))
    old_update = jax.jit(lambda s:f.update(old_model,s,batch))
    for _ in range(2):
        learner,new_metrics = new_update(learner)
        old_learner,old_metrics = old_update(old_learner)
        same(learner,old_learner); same(new_metrics,old_metrics)
    # Real collector, complete-episode ring and RNG, including actual terminal resets.
    old_env,old_state = f.e.initial_state(old_model,old_learner,9,capacity=128)
    _,new_state = ac.initial_state(model,learner,9,capacity=128)
    old_collect = jax.jit(lambda s:f.e.collect(old_model,old_env,s,uniform=True)[0])
    new_collect = jax.jit(lambda s:ac.collect(model,env,s,uniform=True)[0])
    terminals = 0
    for _ in range(55):
        old_state,new_state = old_collect(old_state),new_collect(new_state)
        same(old_state.pilot,new_state)
        if int(new_state.env_state.timestep) == 0:
            terminals += 1
            assert not np.any(new_state.actor_carry) and not np.any(new_state.previous_action)
    assert terminals > 0
    same(f.e.replay.sample(old_state.pilot.replay,jax.random.PRNGKey(19),128,25),
         ac.replay.sample(new_state.replay,jax.random.PRNGKey(19),128,25),exact=True)
    old_keys = f.evaluation_keys(9)
    same(ac.evaluation_keys(9),(jnp.array(old_keys["reset_keys"],jnp.uint32),jnp.array(old_keys["action_keys"],jnp.uint32)),exact=True)
    keys = ac.evaluation_keys(9,3)
    same(ac.evaluate(model,env,learner.online,*keys),f.e.pfe.evaluate(old_model,old_env,old_learner.online,*keys),exact=True)
    # Actual collect-one/update-one production entrypoints at B128/T25 (width8 fixture).
    old_step = jax.jit(lambda s:f.training_step(old_model,old_env,s))
    new_step = jax.jit(lambda s:ac.training_step(model,env,s))
    for _ in range(2):
        old_state,old_metrics = old_step(old_state)
        new_state,new_metrics = new_step(new_state)
        same(old_state.pilot,new_state); same(old_metrics,new_metrics)


def test_production_width_parity():
    """Full 512-wide architecture, T25/B2 short neural comparison; no training run."""
    study = os.environ.get("AC_LAMBDA_STUDY")
    if not study:
        pytest.skip("set AC_LAMBDA_STUDY for active full-width oracle")
    sys.path.append(study)
    f = importlib.import_module("experiments.battleship_ladder.battleship_fe_addons")
    cm,core = f.e.five.initialize_model(9)
    old_model,old = f.adapt(cm,core,f.Config("F5-lambda-off"))
    model,new = ac.initialize_model(9)
    same(old,new,exact=True)
    # Two deterministic complete histories: nonunit behavior ratios and padding.
    t = jnp.arange(25)[:,None]
    lengths = jnp.array([7,11])
    valid = t < lengths
    actions = jnp.broadcast_to(t,(25,2)).astype(jnp.int32)
    legal = jnp.arange(25)[None,None,:] >= t[...,None]
    legal = jnp.broadcast_to(legal,(25,2,25))
    obs = jnp.concatenate((jnp.broadcast_to((t%2)[...,None],(25,2,1)),legal),-1).astype(jnp.float32)
    done = t == lengths-1
    batch = ac.replay.EpisodeBatch(obs,actions,jnp.where(done,25.,-1.),done,
        jnp.full((25,2),.2),jnp.zeros((25,2),jnp.int32),jnp.zeros((25,2),jnp.int32),
        jnp.broadcast_to(t,(25,2)),jnp.broadcast_to(t,(25,2)),valid,jnp.array(True))
    same(ac.targets(model,new.target,batch)[0],f.e.qc.control_targets(old_model,old.target,batch,"QC-qtarget")[0])
    ng = jax.jit(jax.grad(lambda p:ac.objectives(model,p,new.target,batch)[0].sum()))(new.online)
    og = jax.jit(jax.grad(lambda p:f.objectives(old_model,p,old.target,batch)[0].sum()))(old.online)
    same(ng,og)
    nu = jax.jit(lambda s:ac.update(model,s,batch))
    ou = jax.jit(lambda s:f.update(old_model,s,batch))
    for _ in range(2):
        new,nm = nu(new)
        old,om = ou(old)
        same(new,old); same(nm,om)
