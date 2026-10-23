"""外部 SNR 数据扩展点，不从 TSRD 幅度推断信噪比。"""
from pathlib import Path
import numpy as np
from data.loader import Sample


def load_snr_pdw(path, expected_snr_db=None):
    """NPZ 契约：pdw[N,5], labels[N], snr_db 标量；可选 parent[N]。

    IQ 需另附采样率、脉冲边界及与 PDW 的对应关系，经检测器提取后再使用此入口。
    """
    with np.load(Path(path),allow_pickle=False) as f:
        snr=float(f["snr_db"])
        if expected_snr_db is not None and not np.isclose(snr,expected_snr_db):
            raise ValueError("SNR 标注与实验条件不一致")
        x=f["pdw"].copy(); y=f["labels"].copy()
        parent=f["parent"].copy() if "parent" in f else np.arange(len(y))
    return Sample(x,y,parent,{"file":str(Path(path).resolve()),"snr_db":snr,"source":"external_measured_snr"}).sorted()
