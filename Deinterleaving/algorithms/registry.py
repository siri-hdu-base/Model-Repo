"""25 项注册表；实现成熟度和适用条件随结果保存，不能用名字伪装严格复现。"""
import numpy as np
from algorithms.classical import CLUSTERING,TEMPORAL,cluster_predict,temporal_predict
from algorithms.paper_temporal import RequiredPrior

PAPERS={
    "edpc":("Adaptive Radar Signal Sorting via Enhanced Density Peak Clustering","paper_implementation_with_disclosed_numerical_choices","main"),
    "hypergraph":("A Hypergraph-Based Multifunction Radar Signal Sorting Method for Mitigating Batch-Increasing in Complex Electromagnetic Environments","paper_implementation_pending_ambiguity_resolution","main"),
    "sapd":("Deinterleaving pulse trains in unconventional circumstances using multiple hypothesis tracking algorithm","paper_implementation_with_resource_limits","main"),
    "dwell_switch":("Deinterleaving Method for Radar Signals With Dwell and Switch Pulse Repetition Intervals in a Prior Unknown Scenario","paper_implementation_pending_search_validation","main"),
    "dae":("Deinterleaving of Pulse Streams With Denoising Autoencoders","requires_target_prior_and_checkpoint","prior"),
    "incremental":("Incremental Deinterleaving of Radar Emitters","leader_algorithm_2_with_feature_adapter","main"),
    "nmt":("Model-Based Representation and Deinterleaving of Mixed Radar Pulse Sequences With Neural Machine Translation Network","requires_consistent_label_vocabulary_and_checkpoint","prior"),
    "pcid":("Radar Signal Deinterleaving With Progressive Correlation-Integrated Neural Network","requires_waveform_correlation_and_architecture_verification","prior"),
    "cascaded":("Radar Signal Sorting via Cascaded Pulse Parameters","paper_implementation_pending_threshold_resolution","main"),
    "grammar":("Sequential extraction and recognition of pulse group structure for multi-function radar","requires_group_grammar_prior_and_native_validation","prior"),
}
ALL=(*TEMPORAL,*CLUSTERING,"gru","tcn","transformer",*PAPERS)


def information(name):
    if name in PAPERS:
        title,fidelity,track=PAPERS[name]
        return {"title":title,"fidelity":fidelity,"track":track,"strict_reproduction_verified":False}
    return {"title":name,"fidelity":"benchmark_baseline","track":"main","strict_reproduction_verified":None}


def predict(name,x,profile,params,seed):
    """无 y、parent、场景名称或真实源数参数。"""
    if name not in ALL:raise ValueError(f"未知算法 {name}")
    if params.get("strict_papers_only",False) and name in PAPERS:
        raise RequiredPrior("当前论文实现未完成独立严格复现验收，strict_papers_only 禁止输出评分")
    if name in ("dae","nmt","pcid"):
        from algorithms.neural import paper_neural_predict
        return paper_neural_predict(name,x,params)
    if name=="grammar":
        from algorithms.paper_temporal import grammar_extract
        return grammar_extract(x,params)
    if not len(x):return np.empty(0,dtype=np.int64)
    if name in CLUSTERING:return cluster_predict(name,x,profile,params,seed)
    if name in TEMPORAL:return temporal_predict(name,x,params)
    if name in ("gru","tcn","transformer"):
        from algorithms.neural import embedding_predict
        return embedding_predict(name,x,profile,params)
    if name in ("edpc","hypergraph","cascaded","incremental"):
        from algorithms import paper_spatial as p
        if name=="hypergraph":return p.hypergraph(x,params,seed)
        if name=="incremental":return p.incremental(x,profile,params)
        return getattr(p,name)(x,params)
    from algorithms import paper_temporal as p
    return getattr(p,name)(x,params)
