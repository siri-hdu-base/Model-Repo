"""独立的空间聚类和 TOA 方法。参数范围属于可配置基线，不冒充论文最优参数。"""
import numpy as np
from scipy.signal import find_peaks,fftconvolve
from sklearn.cluster import KMeans,DBSCAN,HDBSCAN,MeanShift,SpectralClustering,AgglomerativeClustering,estimate_bandwidth
from data.loader import pdw_features

CLUSTERING=("kmeans","dbscan","hdbscan","mean_shift","spectral","hierarchical")
TEMPORAL=("cdif","sdif","pri_transform","hough","search","correlation")


def spatial(x,profile):
    # 使用 RF、PW、圆周 AOA；幅度可受天线扫描影响，默认不作为空间聚类特征。
    return pdw_features(x,profile)[:,:4]


def estimated_k(z,params):
    labels=HDBSCAN(min_cluster_size=max(2,min(len(z),params.get("min_cluster_size",8))),min_samples=min(len(z),params.get("min_samples",4)),copy=True).fit_predict(z)
    return max(1,len(np.unique(labels[labels>=0])))


def cluster_predict(name,x,profile,params,seed):
    z=spatial(x,profile);n=len(z)
    if n<2: return np.full(n,-1,dtype=int)
    min_size=max(2,min(n,int(params.get("min_cluster_size",8))))
    if name=="dbscan": model=DBSCAN(eps=params.get("eps",.12),min_samples=params.get("min_samples",4))
    elif name=="hdbscan": model=HDBSCAN(min_cluster_size=min_size,min_samples=min(n,params.get("min_samples",4)),copy=True)
    elif name=="mean_shift":
        bw=params.get("bandwidth") or max(estimate_bandwidth(z,quantile=.15,n_samples=min(500,n),random_state=seed),1e-5)
        model=MeanShift(bandwidth=bw,bin_seeding=False)
    else:
        k=estimated_k(z,params)
        if name=="kmeans": model=KMeans(n_clusters=k,n_init=10,random_state=seed)
        elif name=="spectral":
            if k==1:return np.zeros(n,dtype=int)
            model=SpectralClustering(n_clusters=k,affinity="nearest_neighbors",n_neighbors=min(10,n-1),assign_labels="kmeans",random_state=seed)
        elif name=="hierarchical": model=AgglomerativeClustering(n_clusters=k,linkage="ward")
        else: raise ValueError(name)
    return model.fit_predict(z).astype(np.int64)


def candidate_pris(t,name,params):
    """CDIF 累积差分/SDIF 逐阶差分；PRI 变换保留到达时刻复相位。"""
    n=len(t);low=params.get("pri_min_us",2.);high=params.get("pri_max_us",5000.);bw=params.get("bin_us",2.)
    bins=int(np.ceil((high-low)/bw))+1
    if bins>100000: raise ValueError("PRI 网格过密，请显式调整 bin_us")
    counts=np.zeros(bins); phase=np.zeros(bins,dtype=complex)
    order=min(params.get("max_order",30),n-1)
    for lag in range(1,order+1):
        d=t[lag:]-t[:-lag];ok=(d>=low)&(d<=high)
        ix=np.rint((d[ok]-low)/bw).astype(int)
        hist=np.bincount(ix,minlength=bins)
        if name=="sdif":
            # SDIF 各阶独立阈值，检测成功的峰进入候选集。
            threshold=max(params.get("min_chain",5),params.get("threshold_sigma",3.)*np.sqrt(hist.mean()+1))
            counts+=np.where(hist>=threshold,hist,0)
        else: counts+=hist
        if name=="pri_transform":
            tau=low+ix*bw
            np.add.at(phase,ix,np.exp(2j*np.pi*t[lag:][ok]/tau))
    if name=="pri_transform": counts=np.abs(phase)
    if name=="correlation":
        # 有界栅格的自相关，不创建长达数千万微秒的无限高分辨率序列。
        dt=max(bw,(t[-1]-t[0])/params.get("max_time_bins",100000))
        idx=np.floor((t-t[0])/dt).astype(int)
        train=np.bincount(idx).astype(float)
        corr=fftconvolve(train,train[::-1],mode="full")[len(train)-1:]
        grid=low+np.arange(bins)*bw
        counts=np.interp(grid/dt,np.arange(len(corr)),corr,left=0,right=0)
    if name=="hough":
        # t = phase + k*PRI 的周期直线族，在 PRI/phase 平面投票。
        shortlist=np.argsort(counts)[-min(128,len(counts)):]
        votes=np.zeros_like(counts)
        for b in shortlist:
            tau=low+b*bw; phases=np.floor(np.mod(t-t[0],tau)/max(bw,tau/64)).astype(int)
            votes[b]=np.bincount(phases).max(initial=0)
        counts=votes
    peaks,_=find_peaks(counts)
    peaks=np.unique(np.r_[peaks,np.argmax(counts)])
    ranked=peaks[np.argsort(counts[peaks])[::-1]]
    return [float(low+b*bw) for b in ranked[:params.get("max_candidates",64)] if counts[b]>=params.get("min_chain",5)]


def sequence_chain(t,available,tau,tolerance,max_missing):
    """动态规划找最长的近等 PRI 链，允许整数倍间隔表示漏检。"""
    idx=np.flatnonzero(available);a=t[idx];n=len(a)
    length=np.ones(n,dtype=int);previous=np.full(n,-1,dtype=int)
    for j in range(n):
        for m in range(1,max_missing+2):
            target=a[j]-m*tau;pos=np.searchsorted(a,target)
            for q in (pos-1,pos):
                if 0<=q<j and abs(a[q]-target)<=tolerance and length[q]+1>length[j]:
                    length[j]=length[q]+1;previous[j]=q
    if not n:return np.array([],dtype=int)
    j=int(np.argmax(length));chain=[]
    while j>=0:
        chain.append(idx[j]);j=previous[j]
    return np.array(chain[::-1],dtype=int)


def temporal_predict(name,x,params):
    t=x[:,0];labels=np.full(len(t),-1,dtype=int)
    if len(t)<params.get("min_chain",5): return labels
    if name=="search":
        low=params.get("pri_min_us",2.);high=params.get("pri_max_us",5000.);bw=params.get("bin_us",2.)
        # 搜索基线直接从相邻若干对的差值提出候选，不调用 SDIF/CDIF。
        differences=np.concatenate([t[k:]-t[:-k] for k in range(1,min(6,len(t)))])
        bins=np.rint(differences[(differences>=low)&(differences<=high)]/bw).astype(int)
        values,cnt=np.unique(bins,return_counts=True); candidates=(values[np.argsort(cnt)[::-1]][:params.get("max_candidates",64)]*bw).tolist()
    else: candidates=candidate_pris(t,name,params)
    cluster=0
    for tau in candidates:
        tolerance=max(params.get("bin_us",2.),tau*params.get("tolerance",.03))
        for _ in range(params.get("chains_per_pri",8)):
            chain=sequence_chain(t,labels<0,tau,tolerance,params.get("max_missing",3))
            if len(chain)<params.get("min_chain",5):break
            labels[chain]=cluster;cluster+=1
    return labels
