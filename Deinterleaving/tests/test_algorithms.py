import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from algorithms.classical import sequence_chain,CLUSTERING,TEMPORAL
from algorithms.registry import ALL,predict
from algorithms.paper_spatial import natural_neighbors
from algorithms.paper_temporal import maximal_cliques,RequiredPrior
from test_core import PROFILE


def test_registry_has_25():assert len(ALL)==25 and len(set(ALL))==25


def test_chain_missing_pulse():
    t=np.array([0.,30,100,200,240,400,500]);idx=sequence_chain(t,np.ones(len(t),bool),100,1,1)
    np.testing.assert_array_equal(t[idx],[0,100,200,400,500])


def test_clique_is_not_connected_component():
    neighbors={0:{1,2},1:{0,2},2:{0,1,3},3:{2}}
    c=maximal_cliques(neighbors);assert {0,1,2} in c and {2,3} in c and {0,1,2,3} not in c


def test_natural_neighbors_exclude_self():
    d=np.abs(np.arange(6)[:,None]-np.arange(6)[None,:]).astype(float)
    k,knn,mutual,adj=natural_neighbors(d)
    assert not np.diag(mutual).any() and np.all(mutual==mutual.T) and np.all(mutual.sum(axis=1)>0)


@pytest.mark.parametrize('name',CLUSTERING+TEMPORAL)
def test_baseline_observable_separation(name):
    a=np.arange(20)*100.;b=np.arange(20)*170.+17
    x=np.vstack((np.column_stack((a,np.full(20,1000),np.full(20,2),np.zeros(20),np.full(20,-30))),
                 np.column_stack((b,np.full(20,6000),np.full(20,20),np.full(20,90),np.full(20,-60)))))
    x=x[np.argsort(x[:,0])]
    pred=predict(name,x,PROFILE,{'pri_max_us':500,'max_candidates':16,'min_chain':5},0)
    assert pred.shape==(40,) and np.all(pred>=-1)
    assert len(np.unique(pred[pred>=0]))>=1


def test_no_random_weights():
    x=np.zeros((5,5))
    with pytest.raises(RequiredPrior):predict('gru',x,PROFILE,{},0)
    with pytest.raises(RequiredPrior):predict('pcid',x,PROFILE,{},0)


def test_grammar_missing_pulse_and_clutter():
    grammar={'provenance':'independent synthetic test template',
             'subgroups':{'a':{'pri_us':[100.], 'pulse_count':8},'b':{'pri_us':[150.], 'pulse_count':5}},
             'groups':[{'source':0,'subgroups':['a','b'],'gaps_us':[200.]}]}
    times=np.array([0,100,200,250,300,500,600,700,900,1050,1200,1350,1500],float)
    x=np.column_stack((times,np.ones((len(times),4))))
    result=predict('grammar',x,PROFILE,{'grammar':grammar},0)
    assert result[3]==-1 and np.sum(result==0)>=8


@pytest.mark.parametrize('name',['edpc','hypergraph','cascaded','incremental','sapd','dwell_switch'])
def test_paper_modules_do_not_receive_ground_truth(name):
    times=np.arange(20,dtype=float)*100
    x=np.column_stack((times,np.full(20,1000),np.full(20,2),np.zeros(20),np.full(20,-30)))
    y=predict(name,x,PROFILE,{'pri_max_us':500,'pri_step_us':5,'initial_candidates':64},0)
    assert y.shape==(20,) and np.all(y>=-1)


def test_pair_counts_against_bruteforce():
    from evaluation.metrics import evaluate
    true=np.array([0,0,0,1,1,2]);pred=np.array([5,5,-1,6,5,9])
    tp=fp=fn=0
    for i in range(len(true)):
        for j in range(i):
            actual=true[i]==true[j];guessed=pred[i]>=0 and pred[i]==pred[j]
            tp+=int(actual and guessed);fp+=int(not actual and guessed);fn+=int(actual and not guessed)
    result=evaluate(true,pred)
    assert np.isclose(result['pair_f1'],2*tp/(2*tp+fp+fn))
