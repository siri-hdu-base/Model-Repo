"""三个可训练表征基线及论文网络组件。

基线使用样本内同源关系监督，不将 TSRD 文件内的 emitter id 当全局分类标签。
论文分类网络仅允许显式提供跨样本一致的目标标签语义及独立训练权重。
"""
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import HDBSCAN
from data.loader import pdw_features
from algorithms.paper_temporal import RequiredPrior


class TemporalBlock(nn.Module):
    def __init__(self,inputs,width,dilation):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(inputs,width,3,padding=dilation,dilation=dilation),nn.ReLU(),
                               nn.Conv1d(width,width,3,padding=dilation,dilation=dilation),nn.ReLU())
        self.short=nn.Conv1d(inputs,width,1) if inputs!=width else nn.Identity()
    def forward(self,x):return F.relu(self.net(x)+self.short(x))


class EmbeddingNetwork(nn.Module):
    def __init__(self,kind,width=64,embedding=32):
        super().__init__();self.kind=kind;self.width=width
        if kind=="gru":
            self.body=nn.GRU(6,width,num_layers=2,batch_first=True,bidirectional=True);output=width*2
        elif kind=="tcn":
            self.body=nn.Sequential(*[TemporalBlock(6 if i==0 else width,width,2**i) for i in range(6)]);output=width
        elif kind=="transformer":
            self.input=nn.Linear(6,width)
            self.body=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,width*4,dropout=.1,batch_first=True),num_layers=4);output=width
        else:raise ValueError(kind)
        self.projection=nn.Linear(output,embedding)
    def forward(self,x):
        if self.kind=="gru":h,_=self.body(x)
        elif self.kind=="tcn":h=self.body(x.transpose(1,2)).transpose(1,2)
        else:
            h=self.input(x);position=torch.arange(x.shape[1],device=x.device,dtype=x.dtype)[:,None]
            div=torch.exp(torch.arange(0,self.width,2,device=x.device,dtype=x.dtype)*(-math.log(10000)/self.width))
            pe=torch.zeros(x.shape[1],self.width,device=x.device);pe[:,0::2]=torch.sin(position*div);pe[:,1::2]=torch.cos(position*div)
            h=self.body(h+pe)
        return F.normalize(self.projection(h),dim=-1)


def relation_loss(embedding,labels):
    """正负脉冲对分别平均，避免大源和负对数量支配损失。"""
    z=embedding.squeeze(0);labels=labels.reshape(-1)
    distance=(1-z@z.T).clamp_min(0)
    valid=(labels[:,None]>=0)&(labels[None,:]>=0)&torch.triu(torch.ones_like(distance,dtype=torch.bool),diagonal=1)
    positive=valid&(labels[:,None]==labels[None,:]);negative=valid&~(labels[:,None]==labels[None,:])
    if not positive.any() or not negative.any():return None
    return distance[positive].mean()+F.relu(.8-distance[negative]).mean()


_CACHE={}
def embedding_predict(kind,x,profile,params):
    checkpoint=params.get("checkpoint")
    if not checkpoint or not Path(checkpoint).exists():raise RequiredPrior("缺少已训练 checkpoint；请先运行 train，禁止用随机权重评分")
    key=(kind,str(Path(checkpoint).resolve()),params.get("device","cpu"))
    if key not in _CACHE:
        state=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if state.get("kind")!=kind or state.get("mode")!=profile["mode"]:raise ValueError("checkpoint 的算法或接收模式不匹配")
        if state["profile"] != profile:raise ValueError("checkpoint 归一化统计与评估 profile 不一致")
        model=EmbeddingNetwork(kind,**state["architecture"]);model.load_state_dict(state["state_dict"]);model.to(key[2]).eval()
        _CACHE[key]=(model,state)
    model,state=_CACHE[key]
    if params.get("require_formal_checkpoint",False) and state.get("training_scope")!="formal":
        raise RequiredPrior("该权重仅用于 smoke 验证，不能作为正式实验权重")
    z=pdw_features(x,profile,temporal=True).astype(np.float32);emb=[];chunk=params.get("inference_chunk",512)
    with torch.no_grad():
        for start in range(0,len(z),chunk):
            # 重叠上下文只影响特征推理；每个输入脉冲只输出一次。
            left=max(0,start-64);right=min(len(z),start+chunk+64)
            h=model(torch.from_numpy(z[left:right]).unsqueeze(0).to(key[2]))[0].cpu().numpy()
            emb.append(h[start-left:min(start+chunk,len(z))-left])
    if len(z)<2:return np.full(len(z),-1,dtype=int)
    features=np.vstack(emb)
    return HDBSCAN(min_cluster_size=min(len(z),params.get("min_cluster_size",8)),min_samples=min(len(z),params.get("min_samples",4)),copy=True).fit_predict(features)


class NMTNetwork(nn.Module):
    """Zhu 2022：两层 128 单元 BiLSTM，500 单元全连接和 0.25 dropout。"""
    def __init__(self,classes):
        super().__init__();self.recurrent=nn.LSTM(1,128,num_layers=2,bidirectional=True,batch_first=True)
        self.head=nn.Sequential(nn.Linear(256,500),nn.ReLU(),nn.Dropout(.25),nn.Linear(500,classes))
    def forward(self,x):
        h,_=self.recurrent(x);return self.head(h)


class DenoisingAutoencoder(nn.Module):
    """Li 2020：时间二值编码后 128-32-32-128 编解码器。"""
    def __init__(self,bins=3000):
        super().__init__();self.net=nn.Sequential(nn.Linear(bins,128),nn.ReLU(),nn.Linear(128,32),nn.ReLU(),
                                                nn.Linear(32,32),nn.ReLU(),nn.Linear(32,128),nn.ReLU(),nn.Linear(128,bins))
    def forward(self,x):return self.net(x)  # BCEWithLogitsLoss 内部执行 sigmoid。


class CorrelationTransform(nn.Module):
    """PCID Eq.34–37：相关特征生成缩放和偏置，再逐脉冲 LayerNorm。"""
    def __init__(self,width=116):
        super().__init__();self.mod=nn.Sequential(nn.Conv1d(64,width,1),nn.ReLU(),nn.Conv1d(width,2*width,1));self.norm=nn.LayerNorm(width)
    def forward(self,h,c):
        shift,scale=self.mod(c).chunk(2,dim=1)
        return F.relu(self.norm((h*scale+shift).transpose(1,2))).transpose(1,2)


class PCIDNetwork(nn.Module):
    """五维 PDW + 128 维真实波形相关矩阵；网络结构需与论文图2/3逐图核对。"""
    def __init__(self,classes,width=116):
        super().__init__();self.normal=nn.BatchNorm1d(5)
        self.blocks=nn.ModuleList([TemporalBlock(5 if i==0 else width,width,2**i) for i in range(8)])
        self.fusion=nn.ModuleList([CorrelationTransform(width) for _ in range(5)])
        self.attention=nn.ModuleList([nn.Sequential(nn.Conv1d(width,width,3,padding=1),nn.ReLU(),nn.Conv1d(width,width,3,padding=1),nn.Sigmoid()) for _ in range(2)])
        self.head=nn.Conv1d(width,classes,3,padding=1)
    def forward(self,pdw,correlation):
        if correlation.shape[1]!=128:raise ValueError("PCID 需要 128 维波形相关特征")
        near=correlation[:,32:96];far=torch.cat((correlation[:,:32],correlation[:,96:]),dim=1)
        h=self.normal(pdw)
        for i,block in enumerate(self.blocks):
            h=block(h)
            if i<5:h=self.fusion[i](h,near if i<3 else far)
        for attention in self.attention:h=h*attention(h)
        return self.head(h)


def paper_neural_predict(kind,x,params):
    checkpoint=params.get("checkpoint")
    if kind=="pcid" and params.get("correlation_path") is None:
        raise RequiredPrior("PCID Eq.9 要求波形互相关；TSRD PDW 不含该信息")
    if not checkpoint:raise RequiredPrior(f"{kind} 缺少独立训练权重及一致的目标先验；不会随机初始化后评分")
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if not state.get("prior_provenance"):
        raise RequiredPrior("论文 checkpoint 必须包含 prior_provenance 说明标签/目标先验来源")
    with torch.no_grad():
        if kind=="nmt":
            model=NMTNetwork(state["classes"]);model.load_state_dict(state["state_dict"]);model.eval()
            dt=np.diff(x[:,0])/state.get("time_scale_us",1.)
            result=np.full(len(x),-1,dtype=int)
            if len(dt):result[:-1]=model(torch.tensor(dt,dtype=torch.float32)[None,:,None]).argmax(-1)[0].numpy()
            return result  # 论文忽略末脉冲，接口用 -1 显式保留。
        if kind=="dae":
            bins=state.get("bins",3000);unit=state.get("unit_us",10.);model=DenoisingAutoencoder(bins)
            model.load_state_dict(state["state_dict"]);model.eval();result=np.full(len(x),-1,dtype=int)
            # 每个窗口使用同一个已知目标模型，不能把网络输出冒充多个未知源。
            block=np.floor(x[:,0]/(bins*unit)).astype(int)
            for b in np.unique(block):
                mask=np.flatnonzero(block==b);pos=np.floor((x[mask,0]-b*bins*unit)/unit).astype(int)
                vec=np.zeros(bins,np.float32);vec[pos]=1;prob=torch.sigmoid(model(torch.from_numpy(vec)[None]))[0].numpy()
                result[mask[prob[pos]>=.5]]=0
            return result
        if kind=="pcid":
            corr=np.load(params["correlation_path"],allow_pickle=False)
            if corr.shape!=(len(x),128):raise ValueError("相关矩阵与当前样本不对齐")
            model=PCIDNetwork(state["classes"]);model.load_state_dict(state["state_dict"]);model.eval()
            pdw=x[:,[0,2,1,4,3]].copy();pdw[:,0]=np.r_[0,np.diff(x[:,0])]
            return model(torch.tensor(pdw.T,dtype=torch.float32)[None],torch.tensor(corr.T,dtype=torch.float32)[None]).argmax(1)[0].numpy()
    raise ValueError(kind)
