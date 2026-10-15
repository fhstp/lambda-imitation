from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))
import battleship_memory as memory


def test_compact_filters_early_episodes_keeps_post64_and_128_memory():
    import popgym_flash_memory_probe as p
    actions=np.zeros((3,64),int);active=np.ones((3,64),bool);active[1,20:]=False
    hits=np.zeros((3,64),bool);hits[:,0]=True
    labels=p.history_labels(dict(actions=actions,active=active,hits=hits))
    data=dict(memory=np.zeros((3,65,128)),last_input=np.zeros((3,65,65)),actions=actions,outcomes=hits,
        mode=np.array([0,1,2]),split=np.array([0,1,2]),episode_id=np.arange(3),**labels)
    compact,excluded=memory.compact(data)
    assert compact['memory'].shape==(2,128) and excluded==dict(stochastic=0,greedy=1,random=0)
    np.testing.assert_array_equal(compact['episode_id'],[0,2])


def test_reconstruction_fixed_test_selection_and_original_thresholds(tmp_path):
    n=3;valid=np.zeros((n,65),bool);valid[:,64]=True
    data=dict(valid=valid,split=np.full(n,2),episode_id=np.arange(n),mode=np.zeros(n),
        fired=np.zeros((n,65,64),bool),revealed_hits=np.zeros((n,65,64),bool))
    prediction=np.zeros((n,131));prediction[:,3]=.7
    np.savez_compressed(tmp_path/'memory-mlp-heldout.npz',prediction=prediction,threshold=np.full(128,.6),episode_id=np.arange(n))
    rows=memory.reconstruction_rows(data,tmp_path,42,10000)
    assert len(rows)==n*2*64 and rows[0][-1]==1 and rows[0][-2]==.7
    assert rows==memory.reconstruction_rows(data,tmp_path,42,10000)


def test_partial_diagnostic_attempt_is_never_overwritten(tmp_path):
    folder=tmp_path/'diagnostics/step10000/seed42';folder.mkdir(parents=True)
    partial=folder/'partial-proof.txt';partial.write_text('preserve')
    with pytest.raises(FileExistsError,match='Preserved diagnostic attempt'):
        memory.analyze(None,None,{},None,tmp_path,42,10000)
    assert partial.read_text()=='preserve'


def test_episode_auroc_matches_independent_average_rank_reference():
    import popgym_memory_horizon as h
    rng=np.random.default_rng(22)
    scores=rng.integers(0,5,size=(20,64)).astype(float)
    labels=rng.random((20,64))<.2;labels[0]=False;labels[1]=True
    expected=[]
    for score,label in zip(scores,labels):
        positives=int(label.sum());negatives=len(label)-positives
        if not positives or not negatives:expected.append(np.nan);continue
        order=np.argsort(score,kind='stable');ranks=np.zeros(len(score),float)
        for value in np.unique(score):
            positions=np.flatnonzero(score[order]==value)
            ranks[order[positions]]=positions.mean()+1
        expected.append((ranks[label].sum()-positives*(positives+1)/2)/(positives*negatives))
    np.testing.assert_allclose(h.episode_auc(scores,labels),expected,equal_nan=True)


def test_window_hit_target_does_not_refresh_old_hits_with_repeat_misses():
    import popgym_memory_horizon as h
    actions=np.zeros((1,64),int);actions[:,-1]=3
    hits=np.zeros((1,64),bool);hits[:,0]=True
    # Most recent shot at cell0 is a repeat miss;its old hit is outside last5.
    result=h.targets(actions,hits,'window',5)
    assert result[0,0] and result[0,3] and not result[0,64:].any()
    assert h.targets(actions,hits,'window',64)[0,64]
    assert h.targets(actions,hits,'lag',1)[0,3]
    assert h.targets(actions,hits,'lag',64)[0,64]


def test_probability_parity_records_the_reported_one_element_failure_without_raising():
    recorded=np.full((100,64,64),1/64,np.float32);replayed=recorded.copy()
    replayed[2,30,7]+=1.2638047e-5
    p=memory.probability_parity(replayed,recorded,'stochastic',0)
    assert not p['passed'] and p['mismatched_elements']==1 and p['elements']==409600
    assert p['worst_index']==[2,30,7]
    assert p['mismatched_fraction']==1/409600
    assert 1.26e-5<p['max_probability_error']<1.27e-5
    assert p['atol']==1e-5 and p['rtol']==1e-4
    good=memory.probability_parity(recorded,recorded,'random',1)
    summary=memory.parity_summary([p,good])
    assert good['passed'] and summary['warning_blocks']==1 and summary['checked_blocks']==2
    assert summary['mismatched_fraction']==1/819200


def test_nonfinite_and_shape_mismatches_are_still_fatal():
    with pytest.raises(AssertionError,match='shape'):
        memory.probability_parity(np.zeros(3),np.zeros(4),'random',0)
    for bad in (np.nan,np.inf):
        with pytest.raises(FloatingPointError):
            memory.probability_parity(np.array([bad]),np.zeros(1),'random',0)


def test_collection_warns_immediately_and_still_returns_memories(tmp_path,monkeypatch):
    import battleship_sweep
    import popgym_flash_memory_probe as probes
    n,t=100,64
    data=dict(observations=np.zeros((n,t,1)),actions=np.zeros((n,t),int),hits=np.zeros((n,t)),
        history_label_parity=np.ones((n,t),bool),probs=np.full((n,t,64),1/64,np.float32),
        active=np.ones((n,t),bool),rewards=np.zeros((n,t)),done=np.zeros((n,t),bool),q=np.zeros((n,t)),
        repeat=np.zeros((n,t)),seen_mass=np.zeros((n,t)),entropy=np.zeros((n,t)))
    replayed=np.full((n,t+1,64),1/64,np.float32);replayed[0,0,0]+=1.2638e-5
    monkeypatch.setattr(probes.memory,'make_collect',lambda *a:lambda *a,**k:data)
    monkeypatch.setattr(probes,'make_encoder',lambda *a:lambda *a:(np.zeros((n,t+1,2)),replayed,np.zeros((n,t+1,65))))
    monkeypatch.setattr(battleship_sweep,'disk_check',lambda *a:None)
    warnings=[]
    built=(None,dict(memory_hidden_dim=2),None,None,None)
    c=dict(probe_blocks=1,probe_train_blocks=1,probe_val_blocks=1,gamma=.99)
    dataset,parity,summary=memory.collect(None,built,c,tmp_path,False,lambda *args:warnings.append(args))
    assert len(warnings)==3 and len(parity)==3 and dataset['memory'].shape==(300,65,2)
    assert all(p['mismatched_elements']==1 and not p['passed'] for p in parity)
    assert all(w[0]=='policy_memory_parity' for w in warnings)
    assert set(summary)==set(probes.MODES)


@pytest.mark.parametrize('error',[AssertionError('integrity'),FloatingPointError('nonfinite'),
    OSError('disk reserve'),FileExistsError('preserved attempt'),MemoryError('OOM'),SystemError('runtime'),
    type('XlaRuntimeError',(RuntimeError,),{'__module__':'jaxlib.xla_extension'})('device failure')])
def test_optional_wrapper_never_swallows_integrity_resource_or_device_errors(error):
    def fail():raise error
    with pytest.raises(type(error)) as caught:memory.optional_diagnostic('fit',fail)
    assert caught.value is error


def test_only_explicit_optional_stages_can_be_omitted():
    def fail():raise ValueError('probe fit failed')
    with pytest.raises(memory.DiagnosticUnavailable) as caught:memory.optional_diagnostic('probe_fit',fail)
    assert caught.value.stage=='probe_fit' and caught.value.error_type=='ValueError'
    assert memory.optional_diagnostic('fit',lambda:42)==42


def test_parity_statistics_are_logged_to_wandb_as_seed_and_aggregate_scalars(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import wandb
    monkeypatch.setattr(wandb,'Table',lambda **kw:kw)
    monkeypatch.setattr(wandb,'Artifact',lambda *a,**kw:SimpleNamespace(add_file=lambda *a,**kw:None))
    base=tmp_path/'diagnostics/step10000/seed42';base.mkdir(parents=True)
    (base/'reconstructions.json').write_text('[]')
    p=memory.probability_parity(np.array([.0160127]),np.array([.016]),'stochastic',0)
    result=dict(seed=42,full_history={},horizons=[],policy_summary={},extra_diagnostics={},seconds=1.,
        excluded_early_episodes={},certificate=dict(policy_memory_parity=[p]))
    run=SimpleNamespace(id='test',log_artifact=lambda *a,**kw:None)
    payload=memory.log_memory(run,[result],[42],10000,tmp_path,dict(artifact_policy='metrics'))
    assert payload['seed/42/memory/parity/warning_blocks']==1
    assert payload['memory/parity/max_probability_error/mean']==p['max_probability_error']
    assert payload['memory/parity/mismatched_fraction/mean']==1.
    json.dumps({k:v for k,v in payload.items() if isinstance(v,(int,float))},allow_nan=False)
