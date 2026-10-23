"""SAPD 多假设跟踪、D&S 频谱及最大团、先验脉冲组自动机。"""
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks
from algorithms.classical import candidate_pris,sequence_chain


class RequiredPrior(RuntimeError):
    """明确的任务先验缺失，而不是算法返回零分。"""


@dataclass
class Track:
    state: np.ndarray
    covariance: np.ndarray
    score: float
    pulses: tuple
    history: tuple
    misses: int=0


def sapd(x,params):
    """Liu 2010 Eq.1–13 和 N-scan pruning；含漏检虚拟观测分支。"""
    t=x[:,0];n=len(t);out=np.full(n,-1,dtype=int);cluster=0
    f=np.array([[1.,1.],[0.,1.]]);r=params.get("measurement_variance",4.)
    q=np.diag(params.get("process_variance",[1.,1.]));gate=params.get("gate_sigma",3.)
    depth=params.get("n_scan",2);pd=params.get("detection_probability",.9)
    max_hyp=params.get("max_hypotheses",20000);low=params.get("pri_min_us",2.);high=params.get("pri_max_us",5000.)
    attempted=set()
    while True:
        free=np.flatnonzero(out<0);starts=[i for i in free if i not in attempted]
        if not starts:break
        a=starts[0];attempted.add(a);candidates=free[(free>a)&(t[free]-t[a]>=low)&(t[free]-t[a]<=high)]
        if len(candidates)>params.get("initial_candidates",64):
            # 资源限额不能悄悄截断候选，向上层报告。
            raise RuntimeError("resource_limit: SAPD 初始候选超过上限；增加 initial_candidates 后重跑")
        best=None
        for b in candidates:
            hypotheses=[Track(np.array([t[b],t[b]-t[a]]),np.diag([r,2*r]),0.,(a,int(b)),(a,int(b)))]
            finished=[]
            while hypotheses:
                children=[]
                for tr in hypotheses:
                    state=f@tr.state;cov=f@tr.covariance@f.T+q;s=cov[0,0]+r
                    if state[0]>t[-1]+gate*np.sqrt(s) or tr.misses>=params.get("max_missing",10):
                        finished.append(tr);continue
                    delta=gate*np.sqrt(s);i0=np.searchsorted(t,state[0]-delta);i1=np.searchsorted(t,state[0]+delta,side="right")
                    for j in range(i0,i1):
                        if j<=tr.pulses[-1] or out[j]>=0:continue
                        residual=t[j]-state[0];gain=cov[:,0]/s
                        updated=state+gain*residual;pcov=cov-np.outer(gain,cov[0,:])
                        volume=max(high-low,1.)
                        score=tr.score+np.log(volume/np.sqrt(2*np.pi*s))-residual**2/(2*s)
                        children.append(Track(updated,pcov,score,tr.pulses+(j,),tr.history+(j,),0))
                    children.append(Track(state,cov,tr.score+np.log(1-pd),tr.pulses,tr.history+(-1,),tr.misses+1))
                if not children:break
                if len(children)>max_hyp:raise RuntimeError("resource_limit: SAPD 假设数超过上限")
                winner=max(children,key=lambda h:h.score);cut=len(winner.history)-depth
                hypotheses=[h for h in children if h.history[:max(0,cut)]==winner.history[:max(0,cut)]]
            if finished:
                winner=max(finished,key=lambda h:h.score)
                if best is None or winner.score>best.score:best=winner
        if best is not None and len(best.pulses)>=params.get("min_chain",5) and best.score>0:
            out[list(best.pulses)]=cluster;cluster+=1
    return out


def maximal_cliques(neighbors):
    result=[]
    def visit(r,p,x):
        if not p and not x:result.append(r);return
        pivot=max(p|x,key=lambda u:len(p&neighbors[u])) if p|x else None
        for v in sorted(p-(neighbors[pivot] if pivot is not None else set())):
            visit(r|{v},p&neighbors[v],x&neighbors[v]);p.remove(v);x.add(v)
    visit(set(),set(neighbors),set());return result


def dwell_switch(x,params):
    t=x[:,0];n=len(t);out=np.full(n,-1,dtype=int)
    if n<5:return out
    # 原论文把 PRI indicator 当已知分析结果；这里公开使用 CDIF 前端，非真值 PRI。
    candidates=candidate_pris(t,"cdif",params);sequences=[];taus=[];is_ds=[]
    for tau in candidates:
        # Algorithm 1：先找 l1 个连续命中，再向两端扩展 l2 阶搜索视野，最后合并。
        parts=[];pointer=0;tol=2*params.get("toa_error_us",1.);l1=params.get("high_segment_length",3)
        while pointer<n:
            chain=[pointer]
            for _ in range(l1-1):
                target=t[chain[-1]]+tau;pos=np.searchsorted(t,target)
                choices=[j for j in (pos-1,pos) if chain[-1]<j<n and abs(t[j]-target)<=tol]
                if not choices:break
                chain.append(min(choices,key=lambda j:abs(t[j]-target)))
            if len(chain)<l1:pointer+=1;continue
            segment_end=chain[-1]
            for direction in (-1,1):
                while True:
                    endpoint=chain[0] if direction<0 else chain[-1];found=None
                    for multiple in range(1,params.get("search_horizon",3)+1):
                        target=t[endpoint]+direction*multiple*tau;pos=np.searchsorted(t,target)
                        choices=[j for j in (pos-1,pos) if 0<=j<n and direction*(j-endpoint)>0 and abs(t[j]-target)<=tol]
                        if choices:found=min(choices,key=lambda j:abs(t[j]-target));break
                    if found is None:break
                    if direction<0:chain.insert(0,found)
                    else:chain.append(found)
            if len(chain)>=params.get("min_chain",5):parts.extend(chain);pointer=segment_end+1
            else:pointer+=1
        if not parts:continue
        ids=np.array(sorted(set(parts)));a=t[ids];ts=params.get("sample_interval_us",1.)
        size=int(np.ceil((a[-1]-a[0])/ts))+1
        if size>params.get("max_spectrum_bins",1000000):raise RuntimeError("resource_limit: D&S 频谱采样长度超过上限")
        binary=np.zeros(size);binary[np.floor((a-a[0])/ts).astype(int)]=1
        spectrum=np.abs(np.fft.fftshift(np.fft.fft(binary,n=max(16,4*size))))
        freq=np.fft.fftshift(np.fft.fftfreq(max(16,4*size),d=ts))
        principal=spectrum[np.abs(freq)<=1/(2*tau)];principal/=max(principal.max(initial=0),1e-12)
        peaks,_=find_peaks(principal,height=.329)
        sequences.append(ids);taus.append(tau);is_ds.append(len(peaks)>1)
    if not sequences:return out
    def occup(i,left,right,ts):
        a=t[sequences[i]];num=int(np.ceil((right-left)/ts))+2;occ=np.zeros(num,bool);whole=np.zeros(num,bool)
        bins=np.rint((a-left)/ts).astype(int);width=max(1,int(np.rint(taus[i]/ts)))
        for b in bins:occ[b:min(num,b+width+1)]=True
        whole[bins[0]:min(num,bins[-1]+width+1)]=True
        return occ,whole
    neighbors={i:set() for i,v in enumerate(is_ds) if v};cmpt={}
    for i in neighbors:
        for j in neighbors:
            if j<=i:continue
            left=min(t[sequences[i][0]],t[sequences[j][0]])
            right=max(t[sequences[i][-1]]+taus[i],t[sequences[j][-1]]+taus[j]);ts=params.get("sample_interval_us",1.)
            if (right-left)/ts>params.get("max_spectrum_bins",1000000):raise RuntimeError("resource_limit: D&S 区域向量过长")
            a,va=occup(i,left,right,ts);b,vb=occup(j,left,right,ts)
            comp=min(np.sum(va&~a&b)/max(1,va.sum()),np.sum(vb&~b&a)/max(1,vb.sum()))
            excl=max(np.sum(a&b)/max(1,va.sum()),np.sum(a&b)/max(1,vb.sum()))
            cmpt[i,j]=comp
            if comp>.05 and excl<.02:neighbors[i].add(j);neighbors[j].add(i)
    groups=[]
    while neighbors:
        cliques=maximal_cliques(neighbors)
        chosen=max(cliques,key=lambda c:(len(c),sum(cmpt.get(tuple(sorted((i,j))),0) for i in c for j in c if i<j)))
        groups.append(chosen)
        for i in chosen:del neighbors[i]
        for v in neighbors:neighbors[v]-=chosen
    groups += [{i} for i,v in enumerate(is_ds) if not v]
    for c,g in enumerate(groups):
        idx=np.unique(np.concatenate([sequences[i] for i in g]));idx=idx[out[idx]<0];out[idx]=c
    return out


def grammar_extract(x,params):
    """Kang 2022 两层语法的有限展开实现，含启动模板、缺失跃迁、候选消歧。

    grammar.subgroups[id]={pri_us:[...], pulse_count:int};
    grammar.groups=[{source:int, subgroups:[id,...], gaps_us:[...]}].
    先验来自独立资料；source 是先验目标编号，不能来自测试真实标签。
    """
    grammar=params.get("grammar")
    if not grammar or not grammar.get("provenance"):
        raise RequiredPrior("缺少有来源说明的脉冲子组/组语法先验；TSRD 不提供完整目标语法")
    t=x[:,0];out=np.full(len(x),-1,dtype=int);tol=params.get("tolerance_us",2.);templates=[]
    for group in grammar["groups"]:
        offsets=[];states=[];time=0.
        if len(group["gaps_us"])!=len(group["subgroups"])-1:raise ValueError("组间 DTOA 数量错误")
        for g,identifier in enumerate(group["subgroups"]):
            sub=grammar["subgroups"][identifier];pattern=np.asarray(sub["pri_us"],float);count=int(sub["pulse_count"])
            if np.any(pattern<=0) or count<1:raise ValueError("子组 PRI/脉冲数非法")
            for c in range(count):
                offsets.append(time);states.append((g,c,c%len(pattern)))
                if c<count-1:time+=pattern[c%len(pattern)]
            if g<len(group["gaps_us"]):time+=group["gaps_us"][g]
        templates.append((group,np.asarray(offsets),states))
    active=[];completed=[]
    for i,now in enumerate(t):
        survivors=[]
        for state in active:
            template,position,last,matches,origin=state;group,offsets,states=templates[template]
            target=now-origin
            # Eq.8/11 的累积 DTOA 允许越过同子组与相邻子组的缺失脉冲。
            candidates=np.flatnonzero((np.arange(len(offsets))>position)&(np.abs(offsets-target)<=tol))
            if len(candidates):
                for nxt in candidates:
                    updated=(template,int(nxt),i,matches+(i,),origin)
                    if nxt==len(offsets)-1:completed.append(updated)
                    else:survivors.append(updated)
            elif target<=offsets[-1]+tol:survivors.append(state)  # 不匹配的杂波不改变自动机状态。
            elif len(matches)>=4:
                completed.append(state)  # 缺失尾脉冲：在下次启动时消歧。
        # Algorithm 1：对第一子组每个启动位置，至少匹配三个倒序 DTOA。
        for k,(group,offsets,states) in enumerate(templates):
            first_count=grammar["subgroups"][group["subgroups"][0]]["pulse_count"]
            for l in range(3,first_count):
                expected=offsets[l]-offsets[:l];history=t[:i];found=[]
                for delta in expected:
                    target=now-delta;pos=np.searchsorted(history,target)
                    choices=[j for j in (pos-1,pos) if 0<=j<i and out[j]<0 and abs(t[j]-target)<=tol and j not in found]
                    if choices:found.append(min(choices,key=lambda j:abs(t[j]-target)))
                if len(found)>=3:
                    new=(k,l,i,tuple(sorted(found))+(i,),now-offsets[l]);survivors.append(new)
        # 同一模板/起点/状态合并重复预测；不把不同候选随意合并成一个源。
        dedup={}
        for state in survivors:
            key=(state[0],state[1],round(state[4]/max(tol,1e-9),6))
            if key not in dedup or len(state[3])>len(dedup[key][3]):dedup[key]=state
        active=list(dedup.values())
        if len(active)>params.get("max_automata",10000):raise RuntimeError("resource_limit: 语法候选自动机数量超过上限")
    completed.extend(s for s in active if len(s[3])>=4)
    # 只接受唯一 source 解释；不同目标模板共享相同观测则保留为拒识。
    claims={}
    for template,_,_,matches,_ in completed:
        source=int(templates[template][0]["source"])
        for i in matches:claims.setdefault(i,set()).add(source)
    for i,sources in claims.items():
        if len(sources)==1:out[i]=next(iter(sources))
    return out
