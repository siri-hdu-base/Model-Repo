"""六子集读取：HDF5 始终以 r 模式打开，绝不写回原始数据。"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import h5py
import numpy as np

SUBSETS = tuple(f"{m}/{s}" for m in ("scan", "stare") for s in ("train", "val", "test"))
FEATURES = ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"]


def seed_for(*parts):
    """不使用随 Python 进程变化的 hash，保证跨平台种子稳定。"""
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()[:4], "little")


@dataclass
class Sample:
    x: np.ndarray
    y: np.ndarray
    parent: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float64).reshape(-1, 5)
        self.y = np.asarray(self.y, dtype=np.int64).reshape(-1)
        self.parent = np.asarray(self.parent, dtype=np.int64).reshape(-1)
        if not (len(self.x) == len(self.y) == len(self.parent)):
            raise ValueError("PDW、标签、父脉冲编号长度不一致")
        if not np.isfinite(self.x).all():
            raise ValueError("PDW 含 NaN 或无穷值")

    def take(self, index):
        return Sample(self.x[index].copy(), self.y[index].copy(), self.parent[index].copy(), dict(self.meta))

    def sorted(self):
        return self.take(np.argsort(self.x[:, 0], kind="stable"))


def expand_subsets(subsets):
    if isinstance(subsets, str):
        subsets = [subsets]
    result = list(SUBSETS) if "all" in subsets else list(dict.fromkeys(subsets))
    if not result or any(s not in SUBSETS for s in result):
        raise ValueError(f"子集必须属于 {SUBSETS} 或 all")
    return result


def files_for(root, subset, names=None, limit=None):
    mode, split = subset.split("/")
    folder = Path(root) / mode / f"{split}_{mode}"
    paths = sorted(folder.glob("*.h5"), key=lambda p: (int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 10**12, p.name))
    if names:
        wanted = {Path(n).name for n in names}
        paths = [p for p in paths if p.name in wanted]
        missing = wanted - {p.name for p in paths}
        if missing:
            raise FileNotFoundError(f"{subset} 缺少指定文件: {sorted(missing)}")
    if not paths:
        raise FileNotFoundError(f"找不到 HDF5 数据: {folder}")
    return paths if limit is None else paths[:int(limit)]


def feature_order(f):
    """读取实际列名，禁止依靠列位置猜测单位和含义。"""
    key = "metadata/feature_names"
    if key not in f:
        raise ValueError("缺少 metadata/feature_names，需显式适配数据格式")
    names = [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(f[key]).reshape(-1)]
    aliases = {"toa": "ToA", "timeofarrival": "ToA", "frequency": "Frequency", "centrefrequency": "Frequency", "centerfrequency": "Frequency", "rf": "Frequency", "pulsewidth": "PulseWidth", "pw": "PulseWidth", "aoa": "AoA", "angleofarrival": "AoA", "amplitude": "Amplitude"}
    canonical = [aliases.get(n.lower().replace("_", "").replace(" ", ""), n) for n in names]
    if set(canonical) != set(FEATURES):
        raise ValueError(f"未知特征列: {names}")
    return [canonical.index(k) for k in FEATURES]


def inspect_file(path):
    with h5py.File(path, "r") as f:
        order = feature_order(f)
        labels = np.asarray(f["labels"]).reshape(-1)
        n = len(labels)
        if f["data"].shape != (n, 5):
            raise ValueError(f"错误数据维数: {path}")
        # 每个文件只读取时间列与标签；不载入全部五维 PDW。
        times = np.asarray(f["data"][:, order[0]], dtype=np.float64)
        return {"file": str(Path(path).resolve()), "n_pulses": n,
                "n_sources": int(len(np.unique(labels[labels >= 0]))),
                "first_us": float(times.min()) if n else 0., "last_us": float(times.max()) if n else 0.,
                "toa_sorted": bool(np.all(np.diff(times) >= 0)), "shape": list(f["data"].shape),
                "dtype": str(f["data"].dtype), "label_dtype": str(f["labels"].dtype)}


def dense_selection(records, threshold=120):
    """只在调用方指定的单个子集/文件范围内筛选，并保留所有并列最大值。"""
    if not records:
        return [], "empty_subset"
    maximum = max(r["n_sources"] for r in records)
    eligible = [r for r in records if r["n_sources"] >= threshold]
    return (eligible, "at_least_120") if eligible else ([r for r in records if r["n_sources"] == maximum], "subset_maximum_below_120")


def load_window(path, subset, start_us=None, duration_us=None):
    with h5py.File(path, "r") as f:
        order = feature_order(f)
        times = np.asarray(f["data"][:, order[0]], dtype=np.float64)
        start = float(times.min()) if start_us is None and len(times) else float(start_us or 0.)
        stop = start + duration_us if duration_us is not None else float(times.max() + 1) if len(times) else start
        # h5py 对有序连续切片效率更高；异常无序文件使用原始索引筛选。
        if np.all(np.diff(times) >= 0):
            a, b = np.searchsorted(times, [start, stop], side="left")
            idx = np.arange(a, b, dtype=np.int64)
            x = np.asarray(f["data"][a:b], dtype=np.float64)[:, order]
            y = np.asarray(f["labels"][a:b]).reshape(-1)
        else:
            idx = np.flatnonzero((times >= start) & (times < stop))
            x = np.asarray(f["data"][idx], dtype=np.float64)[:, order]
            y = np.asarray(f["labels"][idx]).reshape(-1)
    x[:, 0] -= start
    meta = {"subset": subset, "file": str(Path(path).resolve()), "start_us": start,
            "duration_us": float(stop-start), "units": ["us", "MHz", "us", "degree", "dB"],
            "pri_origin": "synthetic_config_not_inferred_from_scan", "original_count": len(y)}
    return Sample(x, y, idx, meta).sorted()


def window_starts(record, duration_us, count, seed, time_range=None):
    left, right = record["first_us"], record["last_us"]
    if time_range:
        left, right = max(left, time_range[0]), min(right, time_range[1])
    if right < left:
        return []
    if duration_us is None:
        return [left]
    rng = np.random.default_rng(seed)
    return sorted(rng.uniform(left, max(left, right-duration_us), size=count).tolist())


def fit_train_profile(root, mode, max_files=None, rows_per_file=5000):
    """仅从对应 train 拟合物理范围/尺度，采样清单写入 profile。"""
    rows, deltas, sources = [], [], []
    for path in files_for(root, f"{mode}/train", limit=max_files):
        with h5py.File(path, "r") as f:
            order = feature_order(f); n = len(f["data"])
            if n == 0:
                continue
            idx = np.unique(np.linspace(0, n-1, min(n, rows_per_file), dtype=int))
            rows.append(np.asarray(f["data"][idx], dtype=float)[:, order])
            # 连续段时间差只定义测量误差参考尺度，不解释为某个源的 PRI。
            t = np.asarray(f["data"][:min(n, rows_per_file), order[0]], dtype=float)
            d = np.diff(t); deltas.extend(d[d > 0].tolist())
            sources.append(str(path.resolve()))
    if not rows:
        raise ValueError("训练集没有有效 PDW")
    a = np.vstack(rows)
    lo, hi = np.quantile(a, [.001, .999], axis=0)
    center = np.median(a, axis=0)
    scale = np.maximum(np.quantile(a, .75, axis=0)-np.quantile(a, .25, axis=0), [1, 1e-3, 1e-6, 1, 1])
    lo[1:3] = np.maximum(lo[1:3], [1e-6, 1e-9]); hi = np.maximum(hi, lo+1e-6)
    lo[3], hi[3] = -180., 180.
    return {"mode": mode, "fit_subset": f"{mode}/train", "files": sources,
            "rows_per_file": rows_per_file, "center": center.tolist(), "scale": scale.tolist(),
            "low": lo.tolist(), "high": hi.tolist(), "toa_scale_us": float(np.median(deltas)) if deltas else 1.}


def pdw_features(x, profile, temporal=False):
    """角度用正余弦避免 ±180 度不连续；绝对时间不用于空间聚类。"""
    x = np.asarray(x, dtype=float)
    center, scale = np.asarray(profile["center"]), np.asarray(profile["scale"])
    angle = np.deg2rad(x[:, 3])
    z = np.column_stack(((x[:, 1]-center[1])/scale[1], (x[:, 2]-center[2])/scale[2], np.sin(angle), np.cos(angle), (x[:, 4]-center[4])/scale[4]))
    if temporal:
        dtoa = np.r_[0., np.maximum(0, np.diff(x[:, 0]))] if len(x) else np.array([])
        z = np.column_stack((np.log1p(dtoa/max(profile["toa_scale_us"], 1e-9)), z))
    return z
