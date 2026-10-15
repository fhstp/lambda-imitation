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
