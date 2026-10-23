"""单因素扰动与可组合接口。所有变换都复制输入，不修改原始 Sample。"""
import numpy as np
from data.loader import Sample

PRI_MODES = ("jitter", "stagger", "slide", "group", "random", "interleaved", "dwell")
RF_MODES = ("fixed", "simultaneous", "linear", "sawtooth", "random_fixed", "agile")


class MissingScenarioData(RuntimeError):
    pass


def grid():
    specs = [{"name": "baseline"}]
    for name, levels in (("missing", range(10, 100, 10)), ("false_alarm", range(10, 60, 10)),
                         ("noise", range(1, 11)), ("overlap", range(10, 100, 10)), ("split", range(5, 35, 5))):
        specs.extend({"name": name, "level": k/100} for k in levels)
    specs.extend({"name": "pri", "mode": m} for m in PRI_MODES)
    specs.extend({"name": "rf", "mode": m} for m in RF_MODES)
    specs.extend({"name": "short", "level": n} for n in (100, 150, 200, 250, 300))
    specs.append({"name": "dense"})
    specs.extend({"name": "snr", "level": n} for n in (-10, -5, 0, 5, 10))
    return specs


def scenario_id(spec):
    return str(spec["name"]) + ("_"+str(spec.get("mode", spec.get("level"))) if "mode" in spec or "level" in spec else "")


def pri_intervals(mode, n, base, rng, spec):
    i = np.arange(n); table = np.array(spec.get("table", [.8, 1., 1.2])) * base
    if mode == "jitter":
        return base * (1 + rng.uniform(-spec.get("amplitude", .1), spec.get("amplitude", .1), n))
    if mode == "stagger":
        return table[i % len(table)]
    if mode == "slide":
        q = np.minimum(i/max(1, spec.get("ramp_intervals", 64)-1), 1.)
        if spec.get("direction", "up") == "down":
            q = 1-q
        return base*(.8+.4*q)
    if mode == "group":
        return table[(i//spec.get("group_length", 16)) % len(table)]
    if mode == "random":
        return base*rng.uniform(.5, 1.5, n)
    if mode == "dwell":
        out=[]; last=-1
        while len(out) < n:
            choices=[v for v in range(len(table)) if v != last]
            last=int(rng.choice(choices))
            out.extend([table[last]]*int(rng.integers(32, 65)))
        return np.asarray(out[:n])
    raise ValueError(f"未知 PRI 类型 {mode}")


def transform(sample, spec, profile, seed):
    rng = np.random.default_rng(seed)
    s = sample.take(np.arange(len(sample.y)))
    name = spec["name"]; p = float(spec.get("level", 0))
    if name in ("missing", "false_alarm", "noise", "overlap", "split") and not 0 <= p <= 1:
        raise ValueError("比例须在 [0,1] 内")
    real = np.flatnonzero(s.y >= 0); n0 = len(real)
    details = {"scenario": dict(spec), "seed": int(seed), "input_real_count": n0}
    if name == "snr":
        raise MissingScenarioData("缺少真实 SNR 标注的 PDW/IQ，不能由幅度推算 SNR")
    if name == "missing":
        # 四舍五入，不使用 Python 银行家舍入。
        k = int(np.floor(n0*p+.5))
        keep = np.ones(len(s.y), bool); keep[rng.choice(real, k, replace=False)] = False
        details["removed_count"] = k
        details["vanished_sources"] = sorted(set(s.y[real].tolist())-set(s.y[keep & (s.y >= 0)].tolist()))
        s = s.take(keep)
    elif name == "false_alarm":
        k = int(np.floor(n0*p+.5)); lo, hi = np.array(profile["low"]), np.array(profile["high"])
        fake = rng.uniform(lo, hi, (k, 5)); fake[:, 0] = rng.uniform(0, s.meta["duration_us"], k)
        # 负父编号只用于虚警，不与原始编号碰撞。
        first = min(-1, int(s.parent.min())-1) if len(s.parent) else -1
        s = Sample(np.vstack((s.x, fake)), np.r_[s.y, np.full(k, -1)], np.r_[s.parent, first-np.arange(k)], s.meta)
        details["added_count"] = k
    elif name == "noise":
        sigma = np.zeros_like(s.x)
        sigma[:, 0] = profile["toa_scale_us"]*p
        sigma[:, 1:3] = np.abs(s.x[:, 1:3])*p
        sigma[:, 3:] = np.asarray(profile["scale"])[3:]*p
        s.x += rng.normal(size=s.x.shape)*sigma
        before = s.x.copy()
        s.x[:, 0] = np.clip(s.x[:, 0], 0, np.nextafter(s.meta["duration_us"], 0))
        s.x[:, 1:3] = np.maximum(s.x[:, 1:3], [1e-6, 1e-9])
        s.x[:, 3] = (s.x[:, 3]+180) % 360-180
        details["clipped_values"] = int(np.count_nonzero(before[:, :3] != s.x[:, :3]))
    elif name == "overlap":
        labels = np.unique(s.y[real]); centers=[]
        for label in labels:
            a = s.x[s.y == label]
            theta = np.deg2rad(a[:, 3]); angle = np.rad2deg(np.arctan2(np.sin(theta).mean(), np.cos(theta).mean()))
            centers.append([np.median(a[:, 1]), np.median(a[:, 2]), angle])
        if len(labels) < 2:
            details["not_applicable"] = "窗口少于两个源"
        else:
            centers=np.array(centers); scale=np.array(profile["scale"])[[1,2,3]]
            d=centers[:,None,:]-centers[None,:,:]; d[:,:,2]=(d[:,:,2]+180)%360-180
            dist=np.sum((d/scale)**2,axis=2); np.fill_diagonal(dist,np.inf)
            a,b=np.unravel_index(np.argmin(dist),dist.shape)
            delta=centers[b]-centers[a]; delta[2]=(delta[2]+180)%360-180
            for j,sign in ((a,1),(b,-1)):
                mask=s.y==labels[j]; s.x[np.ix_(mask,[1,2,3])] += sign*p*delta/2
            s.x[:,3]=(s.x[:,3]+180)%360-180
            details["source_pair"]=[int(labels[a]),int(labels[b])]
    elif name == "split":
        k=int(np.floor(n0*p+.5)); idx=rng.choice(real,k,replace=False)
        second=s.x[idx].copy(); fraction=rng.uniform(.3,.7,k); original_pw=s.x[idx,2].copy()
        s.x[idx,2]=original_pw*fraction
        second[:,0]+=s.x[idx,2]; second[:,2]=original_pw*(1-fraction)
        s=Sample(np.vstack((s.x,second)),np.r_[s.y,s.y[idx]],np.r_[s.parent,s.parent[idx]],s.meta)
        details["split_parent_count"]=k
        # 以父脉冲起点定义观测范围，允许第二片段延伸到窗口边界外。
        details["window_convention"]="parent_onset; both fragments retained"
    elif name == "short":
        s=s.take(s.x[:,0] < p); s.meta["duration_us"]=p
    elif name == "pri":
        for label in np.unique(s.y[real]):
            idx=np.flatnonzero(s.y==label); count=len(idx)
            base=float(spec.get("base_us",100.))
            if base <= 0:
                raise ValueError("base_us 必须为正")
            phase=rng.uniform(0,base)
            if spec["mode"]=="interleaved":
                a=phase+np.arange((count+1)//2)*base
                b=phase+.4*base+np.arange(count//2)*1.3*base
                times=np.sort(np.r_[a,b])
            else:
                intervals=pri_intervals(spec["mode"],max(0,count-1),base,rng,spec)
                if np.any(intervals<=0):
                    raise ValueError("生成 PRI 必须为正")
                times=phase+np.r_[0,np.cumsum(intervals)]
            s.x[idx,0]=times
        s=s.take(s.x[:,0] < s.meta["duration_us"])
        details["pri_origin"]="synthetic_config"; details["base_us"]=spec.get("base_us",100.)
        details["receiver_model"]="PDW surrogate; scan gating not regenerated"
    elif name == "rf":
        extra=[]; extra_y=[]; extra_p=[]
        for label in np.unique(s.y[real]):
            idx=np.flatnonzero(s.y==label); i=np.arange(len(idx)); center=float(np.median(s.x[idx,1]))
            low=max(1e-6,.9*center); high=min(float(profile["high"][1]),1.1*center)
            high=max(low,high); mode=spec["mode"]; period=int(spec.get("period",64))
            if mode=="fixed": values=np.full(len(idx),center)
            elif mode=="random_fixed": values=np.full(len(idx),rng.uniform(low,high))
            elif mode=="agile": values=rng.uniform(low,high,len(idx))
            elif mode=="linear": values=low+(high-low)*np.minimum(i/max(1,period-1),1)
            elif mode=="sawtooth": values=low+(high-low)*(i%period)/max(1,period-1)
            elif mode=="simultaneous":
                values=np.full(len(idx),low); copies=s.x[idx].copy(); copies[:,1]=high
                extra.append(copies); extra_y.append(s.y[idx]); extra_p.append(s.parent[idx])
            else: raise ValueError(f"未知 RF 类型 {mode}")
            s.x[idx,1]=values
        if extra:
            s=Sample(np.vstack([s.x,*extra]),np.concatenate([s.y,*extra_y]),np.concatenate([s.parent,*extra_p]),s.meta)
    elif name not in ("baseline","dense"):
        raise ValueError(f"未知场景 {name}")
    s.meta={**s.meta,**details,"output_count":len(s.y)}
    return s.sorted()


def compose(sample, specs, profile, seed):
    """按给定顺序串联；每一步的比例分母取该步输入，顺序写入元数据。"""
    out=sample; history=[]
    for i,spec in enumerate(specs):
        out=transform(out,spec,profile,seed+i); history.append(dict(out.meta))
    out.meta["composition"]=history
    return out
