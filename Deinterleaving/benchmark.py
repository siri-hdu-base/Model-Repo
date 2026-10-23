"""命令行入口：inspect / train / evaluate / report / verify-sources。

所有默认路径相对于本工程，不依赖调用命令时的当前工作目录。
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import numpy as np
from data.loader import expand_subsets,fit_train_profile,load_window,window_starts,dense_selection,seed_for
from data.inventory import inventory
from data.scenarios import transform,grid,scenario_id,MissingScenarioData
from algorithms.registry import ALL,information
from evaluation.metrics import evaluate,tsrd_compatibility
from evaluation.runner import IsolatedRunner

BASE=Path(__file__).resolve().parent


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8");temporary.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def output_path(value):
    path=(BASE/value).resolve()
    if not path.is_relative_to(BASE):raise ValueError("输出路径必须位于 Deinterleaving 内，不能写回数据或论文目录")
    path.mkdir(parents=True,exist_ok=True);return path


def read_config(path):
    p=Path(path);p=p if p.is_absolute() else BASE/p
    cfg=json.loads(p.read_text(encoding="utf-8"))
    cfg["data_root"]=str((BASE/cfg["data_root"]).resolve());cfg["subsets"]=expand_subsets(cfg.get("subsets",["scan/test"]))
    cfg["output"]=str(output_path(cfg["output"]))
    if cfg.get("algorithms")=="all":cfg["algorithms"]=list(ALL)
    if any(a not in ALL for a in cfg.get("algorithms",[])):raise ValueError("配置包含未知算法")
    return cfg


def profiles_for(cfg):
    folder=output_path(cfg.get("profile_dir","results/profiles"));result={}
    for mode in sorted({s.split('/')[0] for s in cfg["subsets"]}|set(cfg.get("training",{}).get("modes",[]))):
        path=folder/f"{mode}.json"
        if path.exists():
            profile=json.loads(path.read_text(encoding="utf-8"))
            if profile["fit_subset"]!=f"{mode}/train":raise ValueError("profile 不来自对应训练集")
        else:
            profile=fit_train_profile(cfg["data_root"],mode,cfg.get("profile_files"),cfg.get("profile_rows_per_file",5000));save_json(path,profile)
        result[mode]=profile
    return result


def inventories(cfg):
    result={};folder=output_path(cfg.get("inventory_dir","results/inventory"))
    for subset in cfg["subsets"]:
        names=cfg.get("files",{}).get(subset) if isinstance(cfg.get("files"),dict) else cfg.get("files")
        cache=folder/(subset.replace('/','_')+".json")
        result[subset]=inventory(cfg["data_root"],subset,names,cfg.get("max_files"),cache)
        counts=[r["n_sources"] for r in result[subset]]
        print(f"INSPECT {subset}: {len(counts)} files, maximum sources={max(counts,default=0)}",flush=True)
    return result


def run_evaluation(cfg):
    out=Path(cfg["output"]);rows=out/"rows";rows.mkdir(exist_ok=True)
    arrays=out/"samples";arrays.mkdir(exist_ok=True);predictions=out/"predictions";predictions.mkdir(exist_ok=True)
    profiles=profiles_for(cfg);records=inventories(cfg);settings=cfg.get("algorithm_params",{})
    specs=grid() if cfg.get("scenarios")=="all" else cfg.get("scenarios",[{"name":"baseline"}])
    code_hash=hashlib.sha256(b"".join(p.read_bytes() for p in sorted(BASE.rglob("*.py")) if '.venv' not in p.parts and 'docs' not in p.parts)).hexdigest()
    source_hash={};checkpoints={};model_dir=output_path(cfg.get("checkpoint_dir","results/models"))
    for mode in profiles:
        for algorithm in ("gru","tcn","transformer"):
            for seed in cfg.get("seeds",[0]):
                checkpoint=model_dir/f"{mode}_{algorithm}_seed{seed}.pt"
                if checkpoint.exists():checkpoints[str(checkpoint)]=digest(checkpoint)
    run_config={**cfg,"profiles":profiles,"code_sha256":code_hash,"checkpoint_sha256":checkpoints}
    config_hash=hashlib.sha256(json.dumps(run_config,sort_keys=True).encode()).hexdigest()
    manifest=out/"config_snapshot.json"
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding="utf-8"))
        if old.get("config_sha256")!=config_hash:raise ValueError("输出目录已有不同配置/代码/权重的结果；请更换 output，防止混合实验")
    versions={name:importlib.metadata.version(name) for name in ("numpy","scipy","scikit-learn","h5py","torch","psutil")}
    save_json(manifest,{**run_config,"config_sha256":config_hash,"versions":versions,"python":sys.version})
    runner=IsolatedRunner(cfg.get("timeout_seconds",60),cfg.get("memory_mb",4096),cfg.get("max_pulses",20000))
    counts={};started=time.perf_counter()
    try:
        for subset,items in records.items():
            mode=subset.split('/')[0];dense,dense_kind=dense_selection(items);dense_files={r["file"] for r in dense}
            save_json(out/"selection"/(subset.replace('/','_')+"_dense.json"),{"selection":dense_kind,"records":dense,"scope":"current subset and configured file selection"})
            for record in items:
                if record["file"] not in source_hash:source_hash[record["file"]]=digest(record["file"])
                for seed in cfg.get("seeds",[0]):
                    starts=window_starts(record,cfg.get("window_us",100000.),cfg.get("windows_per_file",1),seed_for(seed,subset,Path(record["file"]).name),cfg.get("time_range_us"))
                    for wi,start in enumerate(starts):
                        base=load_window(record["file"],subset,start,cfg.get("window_us",100000.))
                        for spec in specs:
                            if spec["name"]=="dense" and record["file"] not in dense_files:continue
                            scenario=scenario_id(spec);transform_seed=seed_for(seed,subset,Path(record["file"]).name,wi,spec["name"],spec.get("mode"))
                            # 强度不进入随机种子，尽可能保持场景间配对。
                            token=hashlib.sha256(json.dumps([config_hash,subset,record["file"],source_hash[record["file"]],seed,wi,spec],sort_keys=True).encode()).hexdigest()[:20]
                            try:
                                sample=transform(base,spec,profiles[mode],transform_seed);scenario_error=None
                            except MissingScenarioData as e:sample=None;scenario_error=str(e)
                            if sample is not None:
                                sample.meta.update(record_source_count=record["n_sources"],window_source_count=int(len(np.unique(sample.y[sample.y>=0]))),source_sha256=source_hash[record["file"]])
                                if spec["name"]=="dense":sample.meta["dense_selection"]=dense_kind
                                if not (arrays/(token+".npz")).exists():
                                    np.savez_compressed(arrays/(token+".npz"),pdw=sample.x,labels=sample.y,parent=sample.parent)
                                    save_json(arrays/(token+".json"),sample.meta)
                            for algorithm in cfg["algorithms"]:
                                row_path=rows/f"{token}_{algorithm}.json"
                                if row_path.exists():continue
                                info=information(algorithm)
                                row={"subset":subset,"file":record["file"],"file_sha256":source_hash[record["file"]],"seed":seed,"window":wi,"start_us":start,
                                     "scenario":scenario,"algorithm":algorithm,"sample_id":token,**info,"config_sha256":config_hash}
                                params={**settings.get("common",{}),**settings.get(algorithm,{})}
                                if algorithm in ("gru","tcn","transformer"):
                                    params["checkpoint"]=str(model_dir/f"{mode}_{algorithm}_seed{seed}.pt")
                                    params["device"]=cfg.get("device","cpu");params["require_formal_checkpoint"]=cfg.get("require_formal_checkpoint",False)
                                row["parameters"]=params
                                if scenario_error:row.update(status="not_evaluated",reason=scenario_error)
                                elif sample.meta.get("not_applicable"):row.update(status="not_applicable",reason=sample.meta["not_applicable"])
                                else:
                                    result=runner.run(algorithm,sample.x,profiles[mode],params,seed);labels=result.pop("labels",None);row.update(result)
                                    if labels is not None:
                                        row["metrics"]=evaluate(sample.y,labels,sample.parent)
                                        if cfg.get("tsrd_compatibility",False):row["metrics"].update(tsrd_compatibility(sample.y,labels))
                                        np.save(predictions/f"{token}_{algorithm}.npy",labels,allow_pickle=False)
                                        row["sample_meta"]=sample.meta
                                save_json(row_path,row);counts[row["status"]]=counts.get(row["status"],0)+1
                            print(f"EVAL {subset} {Path(record['file']).name} w={wi} {scenario} {counts}",flush=True)
    finally:
        runner.close();save_json(out/"source_hashes.json",source_hash)
    from evaluation.report import report_run
    report_run(out);save_json(out/"run_summary.json",{"elapsed_seconds":time.perf_counter()-started,"new_status_counts":counts})


def verify_sources():
    manifest=json.loads((BASE/"docs/source_manifest.json").read_text(encoding="utf-8"));changed=[]
    for item in manifest:
        path=Path(item["path"])
        if not path.exists():changed.append({"file":str(path),"reason":"missing"});continue
        st=path.stat()
        if st.st_size!=item["bytes"] or st.st_mtime_ns!=item["mtime_ns"]:changed.append({"file":str(path),"reason":"size_or_mtime"})
        if "sha256" in item and digest(path)!=item["sha256"]:changed.append({"file":str(path),"reason":"sha256"})
    for path in (BASE/"results").rglob("source_hashes.json"):
        for file,value in json.loads(path.read_text(encoding="utf-8")).items():
            if digest(file)!=value:changed.append({"file":file,"reason":"used_data_sha256"})
    result={"protected_file_count":len(manifest),"changed":changed,"verification":"All source size/mtime; all papers and evaluated data SHA256"}
    save_json(BASE/"docs/source_verification.json",result);print(json.dumps(result,ensure_ascii=False),flush=True)
    if changed:raise RuntimeError("源文件发生变化，参见 source_verification.json")


def main():
    parser=argparse.ArgumentParser(description="TSRD 六子集雷达分选 benchmark")
    parser.add_argument("command",choices=["inspect","train","evaluate","report","verify-sources"])
    parser.add_argument("--config",default="configs/smoke.json");parser.add_argument("--subsets",nargs="+")
    parser.add_argument("--output");args=parser.parse_args()
    if args.command=="verify-sources":verify_sources();return
    cfg=read_config(args.config)
    if args.subsets:cfg["subsets"]=expand_subsets(args.subsets)
    if args.output:cfg["output"]=str(output_path(args.output))
    if args.command=="inspect":save_json(Path(cfg["output"])/"inventory.json",inventories(cfg))
    elif args.command=="train":
        from train import train_models
        train_models(cfg,profiles_for(cfg),output_path(cfg.get("checkpoint_dir","results/models")))
    elif args.command=="evaluate":run_evaluation(cfg)
    elif args.command=="report":
        from evaluation.report import report_run
        print(report_run(cfg["output"]))


if __name__=="__main__":
    os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("MKL_NUM_THREADS","1")
    main()
