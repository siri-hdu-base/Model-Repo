"""给论文网络提供独立、显式先验数据的训练入口；不生成TSRD全局源身份。

配置 train_files / val_files 指向 NPZ 文件；NMT: input[N,1], target[N]；
DAE: input[bins], target[bins]；PCID: input[5,N], correlation[128,N], target[N]。
这些已编码样本须由独立训练资料生成，不能使用测试标签冒充已知先验。
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from algorithms.neural import NMTNetwork,DenoisingAutoencoder,PCIDNetwork
from benchmark import BASE,output_path,save_json,digest


def run(config):
    if not config.get("prior_provenance"):raise ValueError("必须提供 prior_provenance")
    kind=config["kind"];device=config.get("device","cpu");torch.manual_seed(config.get("seed",0))
    train=[(BASE/p).resolve() for p in config["train_files"]];val=[(BASE/p).resolve() for p in config["val_files"]]
    if not train or not val or set(train)&set(val):raise ValueError("train/val 必须非空且文件不相交")
    if kind=="nmt":model=NMTNetwork(config["classes"])
    elif kind=="dae":model=DenoisingAutoencoder(config.get("bins",3000))
    elif kind=="pcid":model=PCIDNetwork(config["classes"])
    else:raise ValueError(kind)
    model.to(device);optimizer=torch.optim.Adam(model.parameters(),lr=config.get("learning_rate",.001))
    loss_fn=nn.BCEWithLogitsLoss() if kind=="dae" else nn.CrossEntropyLoss()
    def forward_file(path):
        with np.load(path,allow_pickle=False) as data:
            features=torch.tensor(data["input"],dtype=torch.float32,device=device)[None]
            target=torch.tensor(data["target"],dtype=torch.float32 if kind=="dae" else torch.long,device=device)[None]
            if kind=="pcid":prediction=model(features,torch.tensor(data["correlation"],dtype=torch.float32,device=device)[None])
            else:prediction=model(features)
            if kind=="nmt":prediction=prediction.transpose(1,2)
            return loss_fn(prediction,target)
    best=float("inf");saved=None;history=[];start=time.perf_counter()
    for epoch in range(config.get("epochs",50)):
        model.train();losses=[]
        for path in np.random.default_rng(epoch).permutation(train):
            optimizer.zero_grad();loss=forward_file(path);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():validation=float(np.mean([float(forward_file(p).cpu()) for p in val]))
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"val_loss":validation})
        if validation<best:
            best=validation;saved={**config,"state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"epoch":epoch+1}
    saved["training_seconds"]=time.perf_counter()-start;saved["source_sha256"]={str(p):digest(p) for p in train+val}
    folder=output_path(config.get("output","results/paper_training"));torch.save(saved,folder/(kind+".pt"));save_json(folder/"history.json",history)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);args=parser.parse_args()
    run(json.loads(Path(args.config).read_text(encoding="utf-8")))
