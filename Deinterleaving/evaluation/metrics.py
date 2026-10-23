"""分选评估：不比较任意标签编号，不构造 N×N 脉冲对矩阵。"""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure, fowlkes_mallows_score


def divide(a,b):
    return float(a/b) if b else 0.


def binary_scores(tp,fp,fn,tn=0):
    precision=divide(tp,tp+fp); recall=divide(tp,tp+fn)
    den=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return {"precision":precision,"recall":recall,"f1":divide(2*tp,2*tp+fp+fn),
            "mcc":float((tp*tn-fp*fn)/math.sqrt(den)) if den > 0 else None}


def _match(y,p,w):
    """F1 最优的一对一匹配；预测簇大小包含误吸收虚警，真实拒识计入 FN。"""
    truth=np.unique(y[y>=0]); clusters=np.unique(p[p>=0])
    rt={int(v):i for i,v in enumerate(truth)}; ct={int(v):i for i,v in enumerate(clusters)}
    a=np.zeros((len(truth),len(clusters)),float)
    for yy,pp,ww in zip(y,p,w):
        if yy>=0 and pp>=0: a[rt[int(yy)],ct[int(pp)]]+=ww
    row=np.array([w[y==v].sum() for v in truth]); col=np.array([w[p==v].sum() for v in clusters])
    f=np.divide(2*a,row[:,None]+col[None,:],out=np.zeros_like(a),where=(row[:,None]+col[None,:])>0)
    ri,ci=linear_sum_assignment(-f)
    per=np.zeros(len(truth)); per[ri]=f[ri,ci]
    detected=int(np.count_nonzero(f[ri,ci]>=.5))
    # Accuracy 使用独立的最大正确脉冲数匹配，不能沿用 F1 最优分配。
    ar,ac=linear_sum_assignment(-a)
    total=w[y>=0].sum()
    return {"macro_source_f1":float(per.mean()) if len(per) else None,
            "matched_accuracy":divide(a[ar,ac].sum(),total) if total else None,
            "source_precision":divide(detected,len(clusters)),"source_recall":divide(detected,len(truth)),
            "source_f1":divide(2*detected,len(clusters)+len(truth)),
            "source_count_true":len(truth),"source_count_pred":len(clusters),
            "source_count_abs_error":abs(len(truth)-len(clusters)),
            "split_source_fraction":float(np.mean((a>0).sum(axis=1)>1)) if len(truth) else None,
            "merged_cluster_fraction":float(np.mean((a>0).sum(axis=0)>1)) if len(clusters) else 0.}


def evaluate(y,p,parent=None):
    y=np.asarray(y,dtype=np.int64);p=np.asarray(p,dtype=np.int64)
    if y.shape!=p.shape or y.ndim!=1:
        raise ValueError("预测标签必须是一维且与每个输入 PDW 一一对应")
    if np.any(p < -1): raise ValueError("预测标签必须为 -1 或非负整数")
    real=y>=0; n=int(real.sum())
    result={"n_pulses":len(y),"n_real":n,"n_false_alarm":int((~real).sum()),
            "validity":"empty" if not len(y) else "insufficient_real_pulses" if n<2 else "valid"}
    result.update(_match(y,p,np.ones(len(y))))
    fa=binary_scores(int(np.sum((~real)&(p<0))),int(np.sum(real&(p<0))),int(np.sum((~real)&(p>=0))),int(np.sum(real&(p>=0))))
    result.update({"false_alarm_"+k:(v if np.any(~real) else None) for k,v in fa.items()})
    result["real_rejection_rate"]=divide(np.sum(real&(p<0)),n) if n else None
    for key in ("ari","ami","homogeneity","completeness","v_measure","fmi","pair_precision","pair_recall","pair_f1","pair_mcc"):
        result[key]=None
    if n>=2:
        yt=y[real]; pt=p[real].copy()
        rejected=pt<0; first=int(pt.max(initial=-1))+1
        pt[rejected]=first+np.arange(rejected.sum())
        _,ti=np.unique(yt,return_inverse=True); _,pi=np.unique(pt,return_inverse=True)
        # 稀疏表单元计数，不建立随脉冲数平方增长的矩阵。
        _,cell=np.unique(np.column_stack((ti,pi)),axis=0,return_counts=True)
        row=np.bincount(ti); col=np.bincount(pi)
        choose=lambda v: sum(int(x)*(int(x)-1)//2 for x in v)
        tp=choose(cell); true_pairs=choose(row); pred_pairs=choose(col)
        total=n*(n-1)//2; fp=pred_pairs-tp; fn=true_pairs-tp; tn=total-tp-fp-fn
        pair=binary_scores(tp,fp,fn,tn)
        result.update({"pair_"+k:v for k,v in pair.items()})
        h,c,v=homogeneity_completeness_v_measure(yt,pt)
        result.update(homogeneity=float(h),completeness=float(c),v_measure=float(v),fmi=float(fowlkes_mallows_score(yt,pt)))
        if 0 < true_pairs < total:
            result["ari"]=float(adjusted_rand_score(yt,pt)); result["ami"]=float(adjusted_mutual_info_score(yt,pt))
        else:
            result["validity"]="degenerate_truth_partition"
    if n<2:
        result["macro_source_f1"]=None;result["matched_accuracy"]=None
    if parent is not None and len(y):
        _,inv,counts=np.unique(parent,return_inverse=True,return_counts=True)
        weights=1/counts[inv]
        event=_match(y,p,weights)
        result.update({"parent_equal_"+k:v for k,v in event.items() if k in ("macro_source_f1","matched_accuracy")})
        if n<2:
            result["parent_equal_macro_source_f1"]=None;result["parent_equal_matched_accuracy"]=None
    return result


def tsrd_compatibility(y,p):
    """独立命名：官方 cluster_wise_score 是最差真实簇的最佳一对多匹配，非 pair F1。"""
    from sklearn.metrics import f1_score,matthews_corrcoef
    if len(y)<2 or len(np.unique(y))<2: return {"tdc_min_best_f1":None,"tdc_min_best_mcc":None}
    scores={}
    for name,fun in (("f1",f1_score),("mcc",matthews_corrcoef)):
        vals=[]
        for t in np.unique(y):
            vals.append(max(float(fun(y==t,p==c)) for c in np.unique(p)))
        scores["tdc_min_best_"+name]=min(vals)
    return scores
