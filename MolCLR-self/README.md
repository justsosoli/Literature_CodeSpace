# MolCLR-self

> 基于 **MolCLR**(Molecular Contrastive Learning of Representations via Contrastive Graph Neural Networks)的分子对比学习预训练与下游任务微调实现。

本仓库基于 MolCLR 原论文实现并进行了二次开发,支持 GCN / GIN 两种 GNN 骨干网络、多种分子图数据增强策略(Node Masking / Subgraph Removal / Mix),并提供了在 MoleculeNet 等下游基准上的微调流程。

---

## ✨ 项目特点

- 🔬 **分子图对比预训练**:在 PubChem-10M 等大规模无标签分子数据上进行自监督预训练。
- 🧬 **三种数据增强策略**:Node Masking、Subgraph Removal、Mix。
- 🧠 **多种 GNN 骨干**:GCN、GIN。
- 🎯 **下游任务微调**:BBBP、BACE、ClinTox、Tox21、HIV、SIDER、MUV、FreeSolv、ESOL、Lipo、QM7、QM8、QM9。
- 📊 **TensorBoard 可视化**:训练 loss、学习率、验证指标实时记录。

---

## 📁 项目结构

```text
MolCLR-self/
├── ckpt/                        # 模型检查点与训练日志(TensorBoard)
├── data/                        # 数据集目录
│   ├── pubchem-10m-clean.txt    # 预训练数据集(需自行下载)
│   ├── bbbp/                    # MoleculeNet 下游基准
│   ├── bace/
│   ├── clintox/
│   ├── tox21/
│   ├── hiv/
│   ├── sider/
│   ├── muv/
│   ├── freesolv/
│   ├── esol/
│   ├── lipophilicity/
│   ├── qm7/
│   ├── qm8/
│   └── qm9/
├── datasets/                    # 数据加载与增强
│   ├── dataset.py               # 基础数据集
│   ├── dataset_mix.py           # Mix 增强
│   ├── dataset_subgraph.py      # Subgraph 增强
│   └── dataset_test.py
├── models/                      # GNN 模型定义
│   ├── gcn_model.py             # GCN 预训练模型
│   ├── gin_model.py             # GIN 预训练模型
│   ├── gcn_finetune.py          # GCN 微调模型
│   └── gin_finetune.py          # GIN 微调模型
├── utils/
│   └── nt_xent.py               # NT-Xent 对比损失
├── config.yaml                  # 预训练配置
├── config_finetune.yaml         # 微调配置
├── molclr.py                    # 预训练入口
├── finetune.py                  # 微调入口
├── esol.ipynb                   # 实验 Notebook
├── stu_test.ipynb
└── README.md
```

---

## ⚙️ 环境依赖

项目主要基于 **PyTorch + PyTorch Geometric + RDKit**。

建议使用 **Python 3.8 / 3.9**,并在 Conda 虚拟环境中安装。

```bash
# 推荐使用 conda
conda create -n molclr python=3.9
conda activate molclr

# 安装 PyTorch(以 CUDA 11.8 为例,按需选择)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric 与依赖
pip install torch-geometric==2.4.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# RDKit 与其他依赖
pip install rdkit-pypi networkx numpy pandas pyyaml tensorboard
```

> ⚠️ `torch-scatter` / `torch-sparse` 的版本必须严格匹配本地 PyTorch 与 CUDA 版本,具体见 [PyG 安装文档](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。

---

## 📥 数据准备

### 1. 预训练数据 — PubChem-10M

由于文件较大(`data/pubchem-10m-clean.txt` 通常 ~2GB),默认不包含在仓库中,需要自行下载。

获取方式:

- 官方下载:[https://pubchem.ncbi.nlm.nih.gov/](https://pubchem.ncbi.nlm.nih.gov/)
- 或使用社区清洗后的 10M 子集(参考 MolCLR 官方仓库 README),将文件放置于 `data/pubchem-10m-clean.txt`。

### 2. 下游微调数据 — MoleculeNet

`data/` 目录下的 `bbbp/ bace/ clintox/ tox21/ hiv/sider/ muv/ freesolv/ esol/ lipophilicity/ qm7/ qm8/ qm9/` 等子目录中,每个任务需包含对应的 CSV 文件(如 `BBBP.csv`、`bace.csv` 等)。

可从 [MoleculeNet](https://moleculenet.org/datasets-1) 或 [Chemprop 数据仓库](https://github.com/chemprop/chemprop/tree/master/data) 获取。

---

## 🚀 使用方法

### 1. 自监督预训练

根据需要编辑 `config.yaml`(模型类型、增强策略、数据路径等),然后运行:

```bash
python molclr.py
```

关键参数说明:

- `model_type`:`gcn` / `gin`
- `aug`:`node` / `subgraph` / `mix`
- `dataset.data_path`:预训练数据路径
- `epochs`、`batch_size`、`init_lr` 等按需调整

训练日志与模型会保存到 `ckpt/<时间戳>/`,可用 TensorBoard 查看:

```bash
tensorboard --logdir=ckpt/
```

### 2. 下游任务微调

将预训练得到的模型权重放到 `ckpt/pretrained_<gin/gcn>/model.pth`,然后根据任务修改 `config_finetune.yaml`:

```yaml
task_name: BBBP        # 任务名,可选 BBBP / BACE / ...
fine_tune_from: pretrained_gin
model_type: gin
splitting: scaffold    # random / scaffold
```

启动微调:

```bash
python finetune.py
```

---

## 📈 支持的下游任务

| 任务 | 类型 | 任务 | 类型 |
|------|------|------|------|
| BBBP | 分类 | Tox21 | 分类 |
| BACE | 分类 | HIV   | 分类 |
| ClinTox | 分类 | SIDER | 分类 |
| MUV | 分类 |      |      |
| FreeSolv | 回归 | ESOL | 回归 |
| Lipo(Lipophilicity) | 回归 | QM7 | 回归 |
| QM8 | 回归 | QM9 | 回归 |

---

## 📚 参考资料

- 原始论文:[Molecular Contrastive Learning of Representations via Contrastive Graph Neural Networks](https://arxiv.org/abs/2102.10056)(Wang et al., NeurIPS 2020 Workshop)
- 原作者代码:[shenfenghi/MolCLR](https://github.com/shenfhi/MolCLR)
- MoleculeNet 基准:[Wu et al., MoleculeNet: A Benchmark for Molecular Machine Learning](https://arxiv.org/abs/1703.00564)
- PyTorch Geometric:[https://pyg.readthedocs.io/](https://pyg.readthedocs.io/)

---

## ⚠️ 注意事项

1. **不要上传大型数据与模型权重**:`.gitignore` 已默认忽略 `ckpt/`、`*.pt`、`*.pth` 以及预训练数据 `pubchem-10m-clean.txt`。如确需保留,请使用 [Git LFS](https://git-lfs.github.com/)。
2. **CUDA 版本**:确保 `torch`、`torch-scatter`、`torch-sparse` 三者版本对应,否则可能出现导入错误。
3. **PyCharm 用户**:`.idea/` 已忽略,可安全上传个人配置。
4. **Windows 用户**:`.gitignore` 中的换行符与路径已做 Windows 兼容处理。

---

## 📜 许可证

本项目仅供学习与研究使用,如用于发表请遵循原 MolCLR 仓库及对应数据集的使用条款。