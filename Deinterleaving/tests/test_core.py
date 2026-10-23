"""有确定真值的测试：不只检查函数能否运行。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import h5py
import pytest
from data.loader import Sample,expand_subsets,load_window,dense_selection,SUBSETS
from data.scenarios import transform,pri_intervals,grid,MissingScenarioData
from evaluation.metrics import evaluate

PROFILE={"mode":"scan","fit_subset":"scan/train","low":[0,100,.1,-180,-120],"high":[10000,10000,50,180,-10],
         "center":[0,1000,5,0,-60],"scale":[100,1000,5,60,20],"toa_scale_us":2}


@pytest.fixture
def sample():
    n=100;x=np.column_stack((np.arange(n)*10,np.repeat([1000,2000],50),np.full(n,2.),np.repeat([-179.,179.],50),np.full(n,-50.)))
    return Sample(x,np.repeat([0,1],50),np.arange(n),{"duration_us":10000.})


def test_six_splits():
    assert expand_subsets("all")==list(SUBSETS)
    assert expand_subsets(["scan/test","stare/train"])==["scan/test","stare/train"]
    with pytest.raises(ValueError):expand_subsets("test")


def test_loader_readonly_and_columns(tmp_path):
    path=tmp_path/"a.h5"
    with h5py.File(path,"w") as f:
        f["data"]=np.array([[1,1000,2,0,-20],[2,1001,2,1,-21]],np.float32)
        f["labels"]=np.array([[0],[1]],np.int8)
        f["metadata/feature_names"]=np.array([b'ToA',b'Frequency',b'PulseWidth',b'AoA',b'Amplitude'])
    before=path.read_bytes();s=load_window(path,"scan/test",1,1)
    assert s.x.shape==(1,5) and s.x[0,0]==0 and s.parent.tolist()==[0]
    assert s.x.dtype==np.float64 and s.y.dtype==np.int64 and path.read_bytes()==before


def test_dense_selection():
    records=[{"file":"a","n_sources":90},{"file":"b","n_sources":90},{"file":"c","n_sources":40}]
    selected,kind=dense_selection(records);assert [r['file'] for r in selected]==['a','b'] and 'below_120' in kind
    selected,kind=dense_selection(records+[{"file":"d","n_sources":120}]);assert [r['file'] for r in selected]==['d']


@pytest.mark.parametrize("level",[.1,.5,.9])
def test_missing(sample,level):
    original=sample.x.copy();s=transform(sample,{"name":"missing","level":level},PROFILE,42)
    assert len(s.y)==100-round(100*level) and np.array_equal(original,sample.x)


def test_false_alarm(sample):
    s=transform(sample,{"name":"false_alarm","level":.5},PROFILE,42)
    assert len(s.y)==150 and (s.y==-1).sum()==50
    assert (s.x[:,1:3]>0).all() and len(np.unique(s.parent))==150


def test_split_conservation(sample):
    s=transform(sample,{"name":"split","level":.3},PROFILE,42)
    assert len(s.y)==130
    for i in range(100):
        mask=s.parent==i
        assert np.isclose(s.x[mask,2].sum(),sample.x[i,2])
        assert np.all(s.y[mask]==sample.y[i])
        if mask.sum()==2:
            x=s.x[mask];assert np.isclose(x[1,0],x[0,0]+x[0,2])


def test_overlap_short_arc(sample):
    s=transform(sample,{"name":"overlap","level":.5},PROFILE,42)
    assert np.isclose(np.median(s.x[s.y==0,1]),1250)
    assert np.allclose(s.x[s.y==0,3],-179.5)
    assert np.allclose(s.x[s.y==1,3],179.5)


def test_slide_and_group():
    rng=np.random.default_rng(1);a=pri_intervals("slide",70,100,rng,{})
    np.testing.assert_allclose(a[[0,63,-1]],[80,120,120])
    assert np.all(np.diff(a)>=0)
    b=pri_intervals("group",48,100,rng,{})
    np.testing.assert_array_equal(b,np.repeat([80,100,120],16))


def test_random_and_dwell():
    for name in ("jitter","stagger","slide","group","random","dwell"):
        a=pri_intervals(name,100,100,np.random.default_rng(7),{})
        b=pri_intervals(name,100,100,np.random.default_rng(7),{})
        assert np.array_equal(a,b) and np.all(a>0)
    a=pri_intervals("dwell",100,100,np.random.default_rng(7),{})
    first=np.flatnonzero(a!=a[0])[0];assert 32<=first<=64


def test_simultaneous_parent(sample):
    s=transform(sample,{"name":"rf","mode":"simultaneous"},PROFILE,42)
    assert len(s.y)==200
    for i in range(100):
        a=s.x[s.parent==i];assert len(a)==2 and a[0,0]==a[1,0] and a[0,1]!=a[1,1]


def test_all_scenarios_finite_reproducible(sample):
    for spec in grid():
        if spec['name']=='snr':
            with pytest.raises(MissingScenarioData):transform(sample,spec,PROFILE,5)
            continue
        a=transform(sample,spec,PROFILE,5);b=transform(sample,spec,PROFILE,5)
        np.testing.assert_array_equal(a.x,b.x)
        assert np.isfinite(a.x).all() and np.all(np.diff(a.x[:,0])>=0)


def test_permutation_and_failure_scores():
    y=np.array([0,0,0,1,1,1]);correct=evaluate(y,np.array([9,9,9,3,3,3]))
    assert correct['ari']==1 and correct['macro_source_f1']==1 and correct['matched_accuracy']==1
    rejected=evaluate(y,np.full(6,-1));assert rejected['macro_source_f1']==0 and rejected['real_rejection_rate']==1 and rejected['ari']==0
    merged=evaluate(y,np.zeros(6,dtype=int));assert merged['ari']==0 and merged['macro_source_f1']<1
    fragmented=evaluate(y,np.arange(6));assert fragmented['macro_source_f1']<1


def test_false_alarm_penalizes_cluster_precision():
    y=np.array([0,0,1,1,-1,-1]);a=evaluate(y,np.array([0,0,1,1,-1,-1]));b=evaluate(y,np.array([0,0,1,1,0,0]))
    assert a['macro_source_f1']==1 and b['macro_source_f1']<1
    assert a['false_alarm_f1']==1 and b['false_alarm_recall']==0


def test_degenerate_not_perfect():
    assert evaluate(np.array([],int),np.array([],int))['ari'] is None
    assert evaluate(np.array([0]),np.array([0]))['macro_source_f1'] is None
    assert evaluate(np.array([0,0]),np.array([0,0]))['ari'] is None


def test_parent_equal_weights():
    y=np.array([0,0,0,1,1]);p=np.array([0,0,-1,1,1]);parents=np.array([10,10,11,20,21])
    r=evaluate(y,p,parents)
    assert r['parent_equal_macro_source_f1']<r['macro_source_f1']
