<div align="center">

# 🧪 CHEMIXNET

**基于深度学习的化学分子性质预测框架**
*Deep Learning for Molecular Property Prediction*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-red?logo=keras&logoColor=white)](https://keras.io/)
[![RDKit](https://img.shields.io/badge/RDKit-Chemistry-green)](https://www.rdkit.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

</div>

---

## 📖 项目简介 | Overview

**CHEMIXNET** 是一个面向化学信息学（Cheminformatics）的深度学习实验框架，使用 **RDKit** 处理分子 SMILES 表达，并结合 **TensorFlow / Keras** 构建多种神经网络模型，用于预测化合物在多个化学/生物任务上的性质。

**CHEMIXNET** is a deep learning framework for cheminformatics. It leverages **RDKit** for molecular SMILES processing and **TensorFlow / Keras** to build various neural network architectures for predicting chemical and biological properties of compounds.

### ✨ 主要特性 | Features

- 🧬 **多任务预测**：支持回归（溶解度等）与分类（毒性等）任务
- 🏗️ **多种神经网络架构**：LSTM / GRU / 1D-CNN / 嵌入层等
- 📊 **基准数据集集成**：内置 Tox21、ESOL、FreeSolv、HIV 等常用分子数据集
- 🔬 **基于 RDKit 的特征工程**：支持 QSAR-ready 标准化 SMILES 与分子指纹
- 📓 **可复现的 Notebook 实验**：训练/演化实验以 Jupyter Notebook 形式组织

---

## 📁 项目结构 | Project Structure

```
chemixnet-self/
│
├── data/                        # 数据集（Data sets）
│   ├── toxicity.txt             # Tox21 急性口服毒性数据（带 EPA/GHS 分类）
│   ├── HIV.csv                  # HIV 抗病毒活性数据
│   ├── freesolv.txt             # FreeSolv 溶剂化自由能数据
│   ├── esol.csv                 # ESOL 水溶解度数据
│   └── esol_v2.csv              # ESOL v2 增强版
│
├── evol/                        # 模型训练 / 演化 Notebook
│   ├── evol_tox21.ipynb         # Tox21 毒性预测实验
│   └── evol_esol.ipynb          # ESOL 溶解度预测实验
│
├── util/                        # 工具模块（Utility modules）
│   ├── chemixnet_model_util.py  # 模型构建与编译（Keras / TF）
│   ├── dl_util.py               # 深度学习通用工具
│   ├── eval_util.py             # 评估指标
│   └── rdkit_util.py            # RDKit 分子处理工具
│
├── .gitignore                   # Git 忽略规则
└── README.md                    # 本文件
```

---

## 🚀 快速开始 | Quick Start

### 1. 克隆仓库 | Clone the repository

```bash
git clone https://github.com/<your-username>/CHEMIXNET.git
cd CHEMIXNET
```

### 2. 创建虚拟环境（推荐）| Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. 安装依赖 | Install dependencies

项目主要依赖（建议安装稳定版本）：

```text
# requirements.txt 建议内容
tensorflow>=2.10.0
keras>=2.10.0
rdkit>=2022.9.1
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
jupyter>=1.0.0
matplotlib>=3.6.0
```

安装：

```bash
pip install -r requirements.txt
```

> 💡 **提示**：`rdkit` 推荐通过 `conda install -c conda-forge rdkit` 安装，避免依赖冲突。

### 4. 运行 Notebook | Run the notebooks

```bash
jupyter notebook evol/
```

依次打开 `evol_tox21.ipynb` 或 `evol_esol.ipynb`，按顺序执行单元格即可训练模型并查看结果。

---

## 🧠 模型架构 | Model Architecture

模型定义见 [`util/chemixnet_model_util.py`](util/chemixnet_model_util.py)，主要包含：

| 组件 (Component) | 说明 (Description)                                    |
| ---------------- | ----------------------------------------------------- |
| 嵌入层 Embedding  | 将 SMILES 字符 / token 映射为稠密向量                  |
| LSTM / GRU       | 捕获序列长程依赖                                      |
| 1D-CNN + Pooling | 提取局部结构特征                                      |
| 多输入融合       | `concatenate` 多分支特征融合                          |
| 输出头 Head       | 回归：`linear`；分类：`sigmoid` + `binary_crossentropy` |

超参数（默认值，可在 Notebook 中修改）：

```python
OPTIMIZER    = "adam"
LR           = 1e-3
EMBEDDING_DIM = 128
DROPOUT_RATE  = 0.0
```

---

## 📊 数据集说明 | Datasets

| 数据集 (Dataset)   | 任务类型 (Task)        | 任务描述 (Description)                       |
| ------------------ | --------------------- | -------------------------------------------- |
| `toxicity.txt`     | 分类 (Classification) | Tox21 急性口服毒性 LD50 + EPA/GHS 类别         |
| `HIV.csv`          | 分类 (Classification) | 化合物抑制 HIV 活性                           |
| `freesolv.txt`     | 回归 (Regression)     | 溶剂化自由能                                  |
| `esol.csv`         | 回归 (Regression)     | 水溶解度 (ESOL)                              |
| `esol_v2.csv`      | 回归 (Regression)     | 水溶解度 (ESOL v2)                          |

---

## 📈 评估指标 | Evaluation Metrics

- `Accuracy` / `Precision` / `Recall` — 分类任务
- `AUC` (ROC) — 分类任务
- `MAE` / `MSE` — 回归任务

更多评估实现见 [`util/eval_util.py`](util/eval_util.py)。

---

## 🤝 贡献 | Contributing

欢迎贡献！建议流程：

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/AmazingFeature`
3. 提交修改：`git commit -m "Add some AmazingFeature"`
4. 推送分支：`git push origin feature/AmazingFeature`
5. 提交 Pull Request

---

## 📜 许可证 | License

本项目基于 **MIT License** 开源发布。详见 `LICENSE` 文件。

This project is licensed under the **MIT License**.

---

## 🙏 致谢 | Acknowledgments

- [RDKit](https://www.rdkit.org/) — 开源化学信息学库
- [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/) — 深度学习框架
- 公开数据集：[Tox21](https://ntp.niehs.nih.gov/whatwestudy/tox21), [ESOL](https://moleculenet.ai/datasets-1/), [FreeSolv](https://moleculenet.ai/datasets-1/), [HIV](https://moleculenet.ai/datasets-1/)

---

<div align="center">

⭐ 如果这个项目对您有帮助，欢迎 Star！
*If this project helps you, please consider giving it a star!* ⭐

</div>