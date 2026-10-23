# 雷达信号分选 Benchmark

本工程读取 TSRD 的六个独立子集，在内存中生成场景，运行算法并保存可追溯的结果。原始数据和论文不作修改。

## 当前交付状态

15 个基线包含 6 个 TOA 方法、6 个聚类方法和 GRU、TCN、Transformer。三个深度基线已具有独立的 train/val 训练流程。

10 篇论文均有注册项和对应实现文件，但**注册不等于完成严格复现**。EDPC、超图、SAPD、D&S、增量 leader、级联参数方法可执行；具体数值补全及差异见 `docs/论文实现对应表.md`。DAE、NMT、PCID、层次语法方法需要额外的目标先验、标签语义、权重或波形相关输入。现有 TSRD 条件下明确记为未评估，不能把这些项目视为验收完成。PCID 的网络图核验及部分论文的阈值歧义尚未解决。

正式配置设置 `strict_papers_only=true`，防止尚未通过严格复现核验的论文实现进入正式评分。smoke 配置允许检查可执行论文模块，但结果记录 fidelity，不应宣称为原论文性能。

## 安装与运行

在本目录打开终端。已提供本机独立环境 `.venv`；另一台机器可重新创建环境，安装 `requirements.txt`。GPU 机器需安装与其驱动匹配的 PyTorch CUDA 版本，不复制本机 CPU 虚拟环境。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe benchmark.py inspect --config configs/inspect_all.json
.\.venv\Scripts\python.exe benchmark.py train --config configs/smoke.json
.\.venv\Scripts\python.exe benchmark.py evaluate --config configs/smoke.json
.\.venv\Scripts\python.exe benchmark.py evaluate --config configs/scenarios_smoke.json
.\.venv\Scripts\python.exe benchmark.py evaluate --config configs/scenarios_stare_smoke.json
.\.venv\Scripts\python.exe benchmark.py report --config configs/smoke.json
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe benchmark.py verify-sources
```

## 自由选择数据

六个合法值是 `scan/train`、`scan/val`、`scan/test`、`stare/train`、`stare/val`、`stare/test`。逻辑值自动映射到原始目录，例如 `scan/test_scan`。

```powershell
.\.venv\Scripts\python.exe benchmark.py evaluate --config configs/smoke.json --subsets scan/test --output results/only_scan_test
.\.venv\Scripts\python.exe benchmark.py evaluate --config configs/smoke.json --subsets scan/train --output results/only_scan_train
.\.venv\Scripts\python.exe benchmark.py evaluate --config configs/smoke.json --subsets all --output results/all_six
```

注意：smoke 配置中的 `files` 限定为 `config_1.h5`。若要读取每个子集的全部文件，应删除 `files` 字段或设置为 null，同时将 `max_files` 设为 null。`--subsets all` 只扩展子集，不清除文件限制。

`files` 也可为按子集索引的字典，例如 `{"scan/test":["config_1.h5"],"stare/test":["config_5.h5"]}`。文件不存在时会报错，不静默忽略。

训练由 `training.modes` 控制；选择 train 评估不会触发训练，选择 all 也不会将 test 加入训练。

## 正式实验

修改 `configs/full_gpu.json` 的 `data_root`，按顺序运行 train 和 evaluate。默认正式评估 scan/test、stare/test，每个文件 10 个 100 ms 窗口，三个种子。训练是原始 train 数据，val 选择轮次，绝不使用 test 选权重。

可把 `window_us` 设为 null 评估整条记录。默认脉冲数、内存和时间上限可能触发 resource_limit 或 timeout；应根据资源显式调整，不能把失败样本删除后只报告成功结果。正式训练当前有 max_train_chunks/max_val_chunks 上限，实际用到的样本清单随权重保存。

## 密集多源

在**当前子集和所选文件范围**内统计 HDF5 labels 的真实源数（排除 -1）。存在 ≥120 源记录则全部采用；否则采用源数最多的记录，保留所有并列者。不会复制源、拼接文件或合并六个子集。

结果同时记录原始记录源数和窗口实际源数。若只有一个源或零脉冲，指标按退化规则处理，不自动换成更容易的窗口。

## 结果阅读

每个运行目录含：

- `config_snapshot.json`：配置、代码校验值、软件版本、训练归一化统计和权重校验值。
- `selection/`：每个子集的密集多源选样依据。
- `samples/`：实际测试的派生 PDW、真值、父脉冲编号及变换记录。
- `predictions/`：与样本逐脉冲对齐的预测标签。
- `rows/`、`per_sample.csv`：完整成功及失败记录。
- `summary.csv`、`summary.json`：按文件平均的指标、有效记录数、种子标准差和置信区间。
- `plots/`、`结果说明.md`：场景曲线和运行说明。

低 SNR 无数据时为 not_evaluated；缺先验/权重也是 not_evaluated，原因单独记录。空窗口/单脉冲不会得到虚假的满分。不存在有效重复记录时，不给出虚构置信区间。

代码、配置或权重变化后，请用新的 output 目录。相同快照可断点续跑；不同快照拒绝覆盖混合。

## 工程模块

| 模块 | 职责 |
|---|---|
| data | 只读 HDF5、六子集、源数核验、窗口、场景、SNR 扩展 |
| algorithms | 12 个传统/聚类基线、3 个表征网络、论文组件及注册表 |
| evaluation | 一对一匹配、聚类/脉冲对/虚警指标、隔离执行、汇总绘图 |
| train.py | 仅对应 train 训练，val 选择轮次，保存来源及权重 |
| benchmark.py | inspect/train/evaluate/report/verify-sources 命令 |
| configs | 小规模、场景扫描、全量 GPU 与数据核验配置 |
| docs | 两份 Word 文档、论文对应表、源文件保护与检查记录 |
| tests | 场景守恒、退化指标、算法组件和数据划分验证 |

更多字段、公式和边界约定见两份 Word 文档。

本机验证记录见 `docs/本机验证记录.md`。`scenarios_smoke` 保留 scan 空窗口验证，`scenarios_stare_smoke` 使用相同抽样规则验证 stare 的完整场景网格。它们是功能验证，不作为算法优劣的正式结论。
