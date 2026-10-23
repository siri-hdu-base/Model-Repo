"""论文空间方法。公式编号见 docs/论文实现对应表.md。

本文件保留论文原有样本内 min-max 归一化（算法的一部分），不读取测试真值。
数值容差、未给出的阈值以及退化回退均须在对应表披露。
"""
from collections import deque
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t,skew,kurtosis
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def minmax(a):
    return (a-a.min(axis=0))/np.maximum(np.ptp(a,axis=0),1e-12)


def paper_space(x):
    return minmax(x[:,[1,2,3]])


def natural_neighbors(d,start=1,stable_stop=False):
    n=len(d); ranked=np.argsort(d+np.eye(n)*(d.max()+1),axis=1,kind="stable")[:,:max(0,n-1)]
    old=None
    for k in range(min(start,n-1),n):
        knn=ranked[:,:k]; adjacency=np.zeros((n,n),bool)
        adjacency[np.arange(n)[:,None],knn]=True; mutual=adjacency&adjacency.T
        if np.all(mutual.sum(axis=1)>0) or (stable_stop and old is not None and np.array_equal(old,mutual)):
            return k,knn,mutual,adjacency
        old=mutual.copy()
    return k,knn,mutual,adjacency


def potential_filter(d,params):
    nonzero=d[d>0]
    if not len(nonzero):return np.ones(len(d),bool),np.full(len(d),len(d),dtype=float)
    lower=max(nonzero.min()/10,1e-12);upper=max(nonzero.max()*2,lower*2)
    def objective(log_sigma):
        psi=np.exp(-d*d/(2*np.exp(log_sigma)**2)).sum(axis=1)
        if params.get("exclude_self_potential",False):psi=np.maximum(psi-1,1e-300)
        p=psi/psi.sum()
        return -np.sum(p*np.log10(p))
    optimum=minimize_scalar(objective,bounds=(np.log(lower),np.log(upper)),method="bounded")
    sigma=np.exp(optimum.x);psi=np.exp(-d*d/(2*sigma*sigma)).sum(axis=1)
    if params.get("exclude_self_potential",False):psi=np.maximum(psi-1,0)
    return psi>1+params.get("potential_epsilon",1e-6),psi


def delta_density(d,rho):
    delta=np.full(len(d),d.max());parent=np.full(len(d),-1,dtype=int)
    for i in range(len(d)):
        higher=np.flatnonzero(rho>rho[i])
        if len(higher):
            j=higher[np.argmin(d[i,higher])];delta[i]=d[i,j];parent[i]=j
    return delta,parent


def prediction_upper(x,y):
    design=np.column_stack((np.ones(len(x)),x,x*x)); inv=np.linalg.pinv(design.T@design)
    fit=design@np.linalg.lstsq(design,y,rcond=None)[0]
    dof=max(1,len(x)-np.linalg.matrix_rank(design));variance=np.sum((y-fit)**2)/dof
    leverage=np.einsum("ij,jk,ik->i",design,inv,design)
    return fit+student_t.ppf(.975,dof)*np.sqrt(variance*(1+leverage))


def merge_neighbors(labels,mutual,knn=None,boundary_filter=False,intersection_threshold=0):
    labels=labels.copy();counts=mutual.sum(axis=1)
    while True:
        clusters=np.unique(labels[labels>=0]);changed=False
        unions={}
        for c in clusters:
            mask=labels==c
            if boundary_filter:mask &= counts>=counts[mask].mean()-counts[mask].std()
            unions[c]=set(np.flatnonzero(mutual[mask].any(axis=0)))
        for ia,a in enumerate(clusters):
            for b in clusters[ia+1:]:
                cross=True if knn is None else (np.any(labels[knn[labels==a]]==b) or np.any(labels[knn[labels==b]]==a))
                if cross and len(unions[a]&unions[b])>intersection_threshold:
                    labels[labels==b]=a;changed=True;break
            if changed:break
        if not changed:break
    return labels


def edpc(x,params):
    n=len(x);out=np.full(n,-1,dtype=int)
    if n<4:return out
    d=cdist(paper_space(x),paper_space(x)); keep,_=potential_filter(d,params);idx=np.flatnonzero(keep)
    if len(idx)<4:return out
    d=d[np.ix_(idx,idx)];k,knn,mutual,adj=natural_neighbors(d,stable_stop=True)
    rho=np.exp(-np.take_along_axis(d,knn,axis=1).sum(axis=1)/k)+mutual.sum(axis=1)
    delta,parent=delta_density(d,rho);r=minmax(rho);v=minmax(delta)
    candidate=(v>prediction_upper(r,v))&(r>.5)
    ratio=v/np.maximum(r,1e-12);rank=np.argsort(-ratio,kind="stable")
    upper=prediction_upper(np.arange(len(rank),dtype=float)/len(rank),ratio[rank]);excluded=[]
    for i,j in enumerate(rank):
        if ratio[j]<=upper[i]:break
        excluded.append(j)
    candidate[excluded]=False;centers=np.flatnonzero(candidate)
    # 论文未规定无中心/等密度退化情况：保留最高密度点为可审计数值补全。
    if not len(centers):centers=np.array([np.argmax(rho)])
    labels=np.full(len(idx),-1,dtype=int);labels[centers]=np.arange(len(centers));queue=deque(centers.tolist())
    while queue:
        i=queue.popleft()
        for j in knn[i]:
            if labels[j]<0 and mutual[i,j] and mutual[j].sum()==k:
                labels[j]=labels[i];queue.append(j)
    allocation=np.zeros((len(idx),len(centers)),int)
    for c in range(len(centers)):allocation[:,c]=(labels[knn]==c).sum(axis=1)
    allocation[labels>=0]=0
    while allocation.max(initial=0)>0:
        i,c=np.unravel_index(np.argmax(allocation),allocation.shape);labels[i]=c
        reverse=np.flatnonzero(adj[:,i]&(labels<0));allocation[reverse,c]+=1;allocation[i]=0
    for i in np.argsort(-rho,kind="stable"):
        if labels[i]<0:
            good=np.flatnonzero((labels>=0)&(rho>rho[i]))
            if len(good):labels[i]=labels[good[np.argmin(d[i,good])]]
    labels=merge_neighbors(labels,mutual,boundary_filter=True,intersection_threshold=1)
    out[idx]=labels;return out


def cascaded(x,params):
    n=len(x);out=np.full(n,-1,dtype=int)
    if n<4:return out
    z=paper_space(x);d=cdist(z,z);k,knn,mutual,_=natural_neighbors(d,start=2)
    h=max(np.take_along_axis(d,knn,axis=1)[:,-1].min(),1e-12)
    rho=(np.exp(-np.minimum((d/h)**2,745))*mutual).sum(axis=1)
    rlof=rho[knn].mean(axis=1)/np.maximum(rho,1e-300)
    idx=np.flatnonzero(rlof<=params.get("rlof_threshold",2.))
    if len(idx)<3:return out
    dd=d[np.ix_(idx,idx)];k,knn,mutual,_=natural_neighbors(dd,start=2)
    q1,q3=np.quantile(np.take_along_axis(dd,knn,axis=1)[:,-1],[.25,.75]);eps=max(q3+q3-q1,1e-12)
    labels=DBSCAN(eps=eps,min_samples=k,metric="precomputed").fit_predict(dd)
    labels=merge_neighbors(labels,mutual,knn)
    next_label=0;w=params.get("phase_bins",64);peak_number=params.get("expected_peak_bins",4)
    low=params.get("pri_min_us",2);high=params.get("pri_max_us",5000);step=params.get("pri_step_us",2)
    for c in np.unique(labels[labels>=0]):
        members=idx[labels==c];times=x[members,0];sequences=[]
        for tau in np.arange(low,high+step/2,step):
            bins=np.minimum((np.mod(times,tau)/tau*w).astype(int),w-1);hist=np.bincount(bins,minlength=w)
            if hist.std()==0:continue
            gamma3=hist.mean()/max(float(np.median(hist)),1e-12)
            if skew(hist)>np.sqrt(w/peak_number) and kurtosis(hist)>w/peak_number and gamma3>1:
                peaks=np.flatnonzero((hist>=np.roll(hist,1))&(hist>=np.roll(hist,-1))&(hist>=params.get("min_chain",5)))
                for b in peaks:
                    seq=set(members[bins==b].tolist())
                    if not any(len(seq&a)/len(seq|a)>.9 for a in sequences):sequences.append(seq)
        for seq in sorted(sequences,key=lambda a:(-len(a),min(a))):
            free=np.array(sorted(i for i in seq if out[i]<0),dtype=int)
            if len(free)>=params.get("min_chain",5):out[free]=next_label;next_label+=1
    return out


def fuzzy_cmeans(z,k,rng,iterations=200):
    u=rng.random((len(z),k));u/=u.sum(axis=1,keepdims=True)
    for _ in range(iterations):
        power=u*u;centers=(power.T@z)/np.maximum(power.sum(axis=0)[:,None],1e-12)
        d=np.maximum(cdist(z,centers),1e-10);new=1/(d*d);new/=new.sum(axis=1,keepdims=True)
        if np.max(np.abs(new-u))<1e-5:u=new;break
        u=new
    labels=u.argmax(axis=1);css=float(np.sum((z-centers[labels])**2))
    return u,css


def hypergraph(x,params,seed):
    from scipy import sparse
    n=len(x);out=np.full(n,-1,dtype=int)
    if n<5:return out
    z=paper_space(x);d=cdist(z,z);_,potential=potential_filter(d,{**params,"exclude_self_potential":True})
    # 论文 Eq.5 不含自势能，Eq.6 为 xi 乘全体平均势能。
    threshold=params.get("potential_coefficient",.1)*potential.mean();idx=np.flatnonzero(potential>=threshold)
    if len(idx)<5:return out
    zz=z[idx];dd=d[np.ix_(idx,idx)];m=len(idx);k=min(params.get("neighbors",6),m-1)
    rank=np.argsort(dd+np.eye(m)*(dd.max()+1),axis=1)[:,:k]
    reverse=np.bincount(rank.ravel(),minlength=m);dc=max(dd[np.arange(m),rank[:,0]].max(),1e-12)
    rho=(np.exp(-(dd/dc)**2)*(dd<=dc)).sum(axis=1)-1;delta,_=delta_density(dd,rho)
    core=(delta>delta.mean())&(rho>rho.mean()-rho.std())&(reverse>reverse.mean())
    rng=np.random.default_rng(seed);options=[]
    for c in range(2,min(params.get("max_sources",10),m-1)+1):
        u,css=fuzzy_cmeans(zz,c,rng);lab=u.argmax(axis=1)
        score=silhouette_score(zz,lab) if 1<len(np.unique(lab))<m else -1
        options.append((c,u,css,score))
    # CSS 肘部与 SC 并用：到端点连线距离最大的 c 定义肘部，肘部±1取最高 SC。
    css=np.array([o[2] for o in options]);xs=np.linspace(0,1,len(options));ys=minmax(css) if len(css)>1 else np.ones(1)
    elbow=int(np.argmax((1-xs)-ys));chosen=max(options[max(0,elbow-1):elbow+2],key=lambda o:o[3]);c,u,_,_=chosen
    known=core&(u.max(axis=1)>.5)
    if not known.any():return out  # 不读取真实标签或伪造核心点。
    features=np.column_stack((zz,minmax(potential[idx])));dist=cdist(features,features)
    neighbors=np.argsort(dist,axis=1)[:,:min(k+1,m)]
    rows=neighbors.ravel();cols=np.repeat(np.arange(m),neighbors.shape[1])
    incidence=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(m,m))
    de=np.asarray(incidence.sum(axis=0)).ravel();dv=np.asarray(incidence.sum(axis=1)).ravel()
    normal=sparse.diags(1/np.sqrt(np.maximum(dv,1e-12)))
    transition=normal@incidence@sparse.diags(1/np.maximum(de,1))@incidence.T@normal
    y=np.full((m,c),.5);y[known]=0;y[np.flatnonzero(known),u[known].argmax(axis=1)]=1
    lam=params.get("lambda",.1);alpha=1/(1+lam);f=y.copy()
    for _ in range(params.get("propagation_iterations",500)):
        new=alpha*(transition@f)+(1-alpha)*y
        if np.max(np.abs(new-f))<1e-6:f=new;break
        f=new
    out[idx]=f.argmax(axis=1);return out


def incremental(x,profile,params):
    """Scholl 2023 Algorithm 2：指数更新 leader，50ms 后删除不足六脉冲簇。"""
    from algorithms.classical import spatial
    z=spatial(x,profile);labels=np.full(len(x),-1,dtype=int);centers={};counts={};born={};members={};counter=0
    eps=params.get("eps",.12);alpha=params.get("alpha",.2);ttl=params.get("confirmation_us",50000.)
    for i,point in enumerate(z):
        for c in list(centers):
            if x[i,0]-born[c]>=ttl and counts[c]<6:
                labels[members[c]]=-1;del centers[c]
        keys=list(centers);distance=np.array([np.linalg.norm(point-centers[c]) for c in keys])
        if len(keys) and distance.min()<=eps:
            c=keys[int(distance.argmin())];centers[c]=(1-alpha)*centers[c]+alpha*point
        else:
            c=counter;counter+=1;centers[c]=point.copy();counts[c]=0;born[c]=x[i,0];members[c]=[]
        counts[c]+=1;members[c].append(i);labels[i]=c
    return labels
