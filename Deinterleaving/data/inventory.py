"""读取统计表用于时间范围，源数由原始 labels 核验；缓存只在工程输出目录。"""
from pathlib import Path
import csv
import json
import h5py
import numpy as np
from data.loader import files_for,feature_order,inspect_file


def inventory(root,subset,names=None,limit=None,cache=None):
    paths=files_for(root,subset,names,limit);stats_file=paths[0].parent/"data_stats.csv"
    stats={}
    if stats_file.exists():
        with stats_file.open(encoding="utf-8-sig",newline="") as f:
            stats={str(int(float(r["id"].removeprefix("config_")))):r for r in csv.DictReader(f)}
    previous={}
    if cache and Path(cache).exists():previous=json.loads(Path(cache).read_text(encoding="utf-8"))
    records=[];updated={}
    for path in paths:
        st=path.stat();signature=[st.st_size,st.st_mtime_ns];key=str(path.resolve())
        if key in previous and previous[key].get("signature")==signature:
            record=previous[key]
        else:
            identifier=path.stem.split("_")[-1];row=stats.get(identifier)
            if row is None:record=inspect_file(path)
            else:
                with h5py.File(path,"r") as f:
                    feature_order(f);labels=np.asarray(f["labels"]).reshape(-1)
                    record={"file":key,"n_pulses":len(labels),"n_sources":int(len(np.unique(labels[labels>=0]))),
                            "first_us":float(row["min_ToA"]),"last_us":float(row["max_ToA"]),
                            "shape":list(f["data"].shape),"dtype":str(f["data"].dtype),"label_dtype":str(f["labels"].dtype)}
                    if int(row["n_pulses"])!=len(labels):
                        # 上游统计文件可能过期；不改源文件，以实际 HDF5 为准并保留告警。
                        actual=inspect_file(path)
                        actual["stats_warning"]={"csv_pulses":int(row["n_pulses"]),"hdf5_pulses":len(labels)}
                        record=actual
            record.update(signature=signature,subset=subset,source_count_verified_from_labels=True)
        records.append(record);updated[key]=record
    if cache:
        Path(cache).parent.mkdir(parents=True,exist_ok=True);Path(cache).write_text(json.dumps(updated,ensure_ascii=False,indent=2),encoding="utf-8")
    return records
