"""原始 train 训练、val 选最优轮次；无测试集数据参与。"""
from pathlib import Path
import json
import time
import random
from functools import lru_cache
import numpy as np
import torch
from data.loader import load_window,window_starts,pdw_features,seed_for
from data.inventory import inventory
from algorithms.neural import EmbeddingNetwork,relation_loss


class WindowChunks:
    """仅保存窗口位置；按需读取，最多缓存两个窗口，不将正式训练集全部放入内存。"""
    def __init__(self,items,profile,window_us,length):
        self.items=items;self.profile=profile;self.window_us=window_us;self.length=length
    def __len__(self):return len(self.items)
    @lru_cache(maxsize=2)
    def window(self,path,subset,start):
        s=load_window(path,subset,start,self.window_us)
        return pdw_features(s.x,self.profile,temporal=True).astype(np.float32),s.y
    def __getitem__(self,i):
        entry=self.items[i];x,y=self.window(entry['file'],entry['subset'],entry['start_us']);a=entry['chunk_start']
        return x[a:a+self.length],y[a:a+self.length]


def batches(root,mode,split,settings,profile,seed):
    records=inventory(root,f"{mode}/{split}",limit=settings.get(f"{split}_files"))
    provenance=[];length=settings.get("chunk_pulses",256)
    for record in records:
        starts=window_starts(record,settings.get("window_us",100000.),settings.get("windows_per_file",2),seed_for(seed,Path(record["file"]).name,split))
        for start in starts:
            sample=load_window(record["file"],f"{mode}/{split}",start,settings.get("window_us",100000.))
            features=pdw_features(sample.x,profile,temporal=True).astype(np.float32)
            for begin in range(0,len(features),length):
                y=sample.y[begin:begin+length]
                if len(y)<4 or len(np.unique(y[y>=0]))<2:continue
                provenance.append({"file":record["file"],"subset":f"{mode}/{split}","start_us":start,"chunk_start":begin,"n":len(y)})
                if len(provenance)>=settings.get(f"max_{split}_chunks",100000):
                    return WindowChunks(provenance,profile,settings.get('window_us',100000.),length),provenance
    return WindowChunks(provenance,profile,settings.get('window_us',100000.),length),provenance


def train_models(config,profiles,output):
    settings=config["training"];device=settings.get("device","cpu");torch.set_num_threads(settings.get("threads",2))
    if device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("配置指定 CUDA，但当前环境无可用 GPU；不自动降级")
    report=[];seeds=settings.get("seeds",[0]);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for mode in settings.get("modes",["scan","stare"]):
        profile=profiles[mode]
        for seed in seeds:
            train_data,train_sources=batches(config["data_root"],mode,"train",settings,profile,seed)
            val_data,val_sources=batches(config["data_root"],mode,"val",settings,profile,seed)
            if not train_data or not val_data:raise ValueError(f"{mode} 没有可训练/验证的多源窗口；增加文件或窗口数量")
            for kind in settings.get("algorithms",["gru","tcn","transformer"]):
                if kind not in ("gru","tcn","transformer"):raise ValueError("train 原生入口只训练三个未知源表征基线；论文网络须使用独立先验数据接口")
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                architecture={"width":settings.get("width",64),"embedding":settings.get("embedding",32)}
                model=EmbeddingNetwork(kind,**architecture).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=settings.get("learning_rate",.001),weight_decay=1e-4)
                best=float("inf");history=[];begin=time.perf_counter();saved=None
                for epoch in range(settings.get("epochs",5)):
                    model.train();losses=[];order=np.random.default_rng(seed+epoch).permutation(len(train_data))
                    for j in order:
                        features,labels=train_data[j];features=torch.from_numpy(features).unsqueeze(0).to(device);labels=torch.from_numpy(labels).to(device)
                        optimizer.zero_grad();loss=relation_loss(model(features),labels)
                        if loss is None:continue
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach().cpu()))
                    model.eval();validation=[]
                    with torch.no_grad():
                        for features,labels in val_data:
                            loss=relation_loss(model(torch.from_numpy(features).unsqueeze(0).to(device)),torch.from_numpy(labels).to(device))
                            if loss is not None:validation.append(float(loss.cpu()))
                    if not losses or not validation:raise ValueError("没有同时含正负脉冲对的训练/验证批次")
                    value=float(np.mean(validation));history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"val_loss":value})
                    if value<best:
                        best=value;saved={"kind":kind,"mode":mode,"seed":seed,"architecture":architecture,
                            "state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"profile":profile,
                            "training_scope":settings.get("scope","smoke"),"epoch":epoch+1,"parameter_count":sum(p.numel() for p in model.parameters()),
                            "train_sources":train_sources,"val_sources":val_sources,"config":settings}
                    print(f"TRAIN {mode} {kind} seed={seed} epoch={epoch+1} val={value:.4f}",flush=True)
                saved["training_seconds"]=time.perf_counter()-begin;saved["history"]=history
                destination=output/f"{mode}_{kind}_seed{seed}.pt";torch.save(saved,destination)
                report.append({"checkpoint":str(destination.resolve()),"mode":mode,"algorithm":kind,"seed":seed,"training_scope":saved["training_scope"],
                               "parameter_count":saved["parameter_count"],"training_seconds":saved["training_seconds"],"best_val_loss":best,"history":history})
    (output/"training_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report
