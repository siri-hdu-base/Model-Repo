"""按原始文件平均，再按文件 bootstrap，避免将窗口当独立样本。"""
from pathlib import Path
import csv
import json
from collections import defaultdict,Counter
import numpy as np


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def report_run(folder):
    folder=Path(folder);rows=[json.loads(p.read_text(encoding="utf-8")) for p in sorted((folder/"rows").glob("*.json"))]
    flat=[]
    for row in rows:
        flat.append({k:v for k,v in row.items() if k not in ("metrics","traceback","sample_meta")}|row.get("metrics",{}))
    write_csv(folder/"per_sample.csv",flat)
    groups=defaultdict(list)
    for r in rows:groups[(r["subset"],r["algorithm"],r["scenario"],r["track"],r["fidelity"])].append(r)
    summary=[];rng=np.random.default_rng(20260923)
    for key,group in sorted(groups.items()):
        base=dict(zip(("subset","algorithm","scenario","track","fidelity"),key));base["attempted"]=len(group)
        statuses=Counter(r["status"] for r in group);base["status_counts"]=json.dumps(statuses,ensure_ascii=False)
        successful=[r for r in group if r["status"]=="ok"]
        base["successful"]=len(successful);base["success_rate"]=len(successful)/len(group)
        base["metric_validity_counts"]=json.dumps(Counter(r.get("metrics",{}).get("validity","not_scored") for r in group),ensure_ascii=False)
        names=sorted({k for r in successful for k,v in r.get("metrics",{}).items() if isinstance(v,(float,int)) and not isinstance(v,bool)})
        names+= ["inference_seconds","pulses_per_second","peak_rss_mb"]
        for metric in names:
            by_file=defaultdict(list);by_seed=defaultdict(lambda:defaultdict(list))
            for row in successful:
                value=row.get("metrics",{}).get(metric,row.get(metric))
                if value is not None and np.isfinite(value):
                    by_file[row["file"]].append(float(value));by_seed[row["seed"]][row["file"]].append(float(value))
            values=np.array([np.mean(v) for v in by_file.values()]);base[metric+"_valid_records"]=len(values)
            base[metric+"_mean"]=float(values.mean()) if len(values) else None
            means=[np.mean([np.mean(v) for v in files.values()]) for files in by_seed.values()]
            base[metric+"_seed_std"]=float(np.std(means,ddof=1)) if len(means)>1 else None
            if len(values)>1:
                bootstrap=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(1000)])
                base[metric+"_ci_low"],base[metric+"_ci_high"]=np.quantile(bootstrap,[.025,.975]).tolist()
            else:base[metric+"_ci_low"]=base[metric+"_ci_high"]=None
        summary.append(base)
    write_csv(folder/"summary.csv",summary)
    (folder/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    status=Counter(r["status"] for r in rows)
    text=["# 实际运行结果", "",f"共 {len(rows)} 个算法与样本组合，状态：{dict(status)}。", "",
          "这些结果仅覆盖配置快照中的文件、窗口与种子。smoke 权重是功能验证，不代表充分训练性能。", "",
          "论文实现成熟度随结果保存；未通过严格复现核验的方法不能作为已复现论文主张。缺失、失败和未评估记录不填零分。", "",
          "| 子集 | 算法 | 场景 | 成功数/尝试数 | ARI | 源平均 F1 |", "|---|---|---|---|---|---|"]
    for r in summary:
        fmt=lambda v:"—" if v is None else f"{v:.4f}"
        text.append(f"| {r['subset']} | {r['algorithm']} | {r['scenario']} | {r['successful']}/{r['attempted']} | {fmt(r.get('ari_mean'))} | {fmt(r.get('macro_source_f1_mean'))} |")
    (folder/"结果说明.md").write_text("\n".join(text),encoding="utf-8")
    plots=folder/"plots";plots.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for subset in sorted({r["subset"] for r in summary}):
        for scenario in ("missing","false_alarm","noise","overlap","split","short"):
            for metric in ("ari","macro_source_f1"):
                fig,ax=plt.subplots(figsize=(10,6));found=False
                for algorithm in sorted({r["algorithm"] for r in summary}):
                    points=[r for r in summary if r["subset"]==subset and r["algorithm"]==algorithm and r["scenario"].startswith(scenario+"_") and r.get(metric+"_mean") is not None]
                    if not points:continue
                    points.sort(key=lambda r:float(r["scenario"].rsplit("_",1)[-1]));found=True
                    ax.plot([float(r["scenario"].rsplit("_",1)[-1]) for r in points],[r[metric+"_mean"] for r in points],marker=".",label=algorithm)
                if found:
                    ax.set(title=f"{subset} | {scenario}",xlabel="duration (us)" if scenario=="short" else "fraction",ylabel=metric)
                    ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(plots/f"{subset.replace('/','_')}_{scenario}_{metric}.png",dpi=150)
                plt.close(fig)
    return status
