# 2D 非晶碳 Flow Matching 生成模型 - 实施计划

## 项目概述

**目标**: 基于 E3 等变神经网络架构，构建 Flow Matching 条件生成模型，用于 2D 非晶碳结构生成。

**数据特点**:
- LAMMPS 格式文件 (`.data`)
- 50 个碳原子 / 样本
- 2D 结构 (z = 0)
- 固定盒子: 12×12×20 Å
- 1000 个样本
- 条件: 冷却速率 = [20, 50, 100, 200, 400, 800, 1500, 2500, 5000, 10000] K/ps
- 数据划分: **train:val:test = 8:1:1**

**技术路线**: 多模型后端 (NequIP/EGNN/SchNet) + Flow Matching + PyTorch Lightning

---

## ✅ 阶段一: 环境准备与数据处理 (已完成)

**已创建文件**:
- `diffcsp/pl_data/amorphous_dataset.py` - 数据加载模块

**核心功能**:
1. LAMMPS 数据文件读取 (ASE)
2. 从文件名自动提取冷却速率
3. 冷却速率归一化 (log10 变换 → [0,1])
4. PyG 图构建 (周期性边界)
5. 数据复制增强

**冷却速率提取规则**:
```python
# 文件名: {index}_min.data
# 规则: (index - 1) % 100 // 10 → rate_idx (0-9)
QUENCHING_RATES = [20, 50, 100, 200, 400, 800, 1500, 2500, 5000, 10000]
```

**数据结构**:
```python
data.pos           # (N, 3) 原子位置
data.edge_index    # (2, E) 图连接
data.edge_attr     # (E, 3) 边向量
data.cooling_rate  # log10(rate) - 模型条件
data.quench_rate   # 原始冷却速率 (K/ps)
data.condition     # 归一化到 [0,1]
data.file_index    # 文件索引
```

---

## ✅ 阶段二: Flow Matching 网络改造 (已完成)

**已创建文件**:
- `diffcsp/pl_modules/nequip_flow.py` - NequIP Flow Matching 网络
- `diffcsp/pl_modules/flow_transforms.py` - Flow Matching 变换
- `diffcsp/pl_modules/amorphous_flow_module.py` - Lightning 训练模块
- `diffcsp/pl_modules/model_factory.py` - 模型工厂 (动态模型切换)

### 2.1 多模型后端支持

支持三种 GNN 后端，通过 `model_factory.py` 动态切换：

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| **NequIP** (默认) | E(3) 等变，球谐函数 | 最高精度，几何敏感任务 |
| **EGNN** | E(n) 等变，轻量级 | 平衡速度与精度 |
| **SchNet** | 连续滤波，非等变 | 最快，平滑势能面 |

**使用方式**:
```python
from diffcsp.pl_modules.amorphous_flow_module import AmorphousFlowModule

# 使用 NequIP (默认)
module = AmorphousFlowModule(model_type='nequip')

# 使用 EGNN
module = AmorphousFlowModule(model_type='egnn', model_config={'hidden_dim': 256})

# 使用 SchNet
module = AmorphousFlowModule(model_type='schnet')
```

### 2.2 Flow Matching 实现

**线性插值 (OT-CFM)**:
```python
x_t = (1-t) * x_0 + t * x_1  # x_0: 噪声, x_1: 数据
v_target = x_1 - x_0         # 目标速度场
```

**损失函数**:
```python
loss = MSE(v_pred, v_target)
```

### 2.3 条件生成

- 条件: 冷却速率 (log10 变换)
- 嵌入方式: 正弦位置编码 + MLP
- 注入位置: 节点特征 + 每个卷积层

**已验证**:
- ✅ 前向传播成功
- ✅ 损失计算正确
- ✅ 反向传播正常
- ✅ 采样生成有效
- ✅ 多模型切换正常

---

## ✅ 阶段三: 训练框架集成 (已完成)

### 3.1 配置文件

**已创建文件**:
- `conf/amorphous_flow.yaml` - 主配置入口
- `conf/data/amorphous_carbon.yaml` - 数据配置
- `conf/model/amorphous_flow.yaml` - NequIP 模型配置
- `conf/model/amorphous_flow_egnn.yaml` - EGNN 模型配置
- `conf/model/amorphous_flow_schnet.yaml` - SchNet 模型配置
- `conf/logging/amorphous_flow.yaml` - W&B 日志配置
- `conf/train/amorphous_flow.yaml` - 训练配置

### 3.2 训练脚本

**已创建文件**: `diffcsp/train_amorphous.py`

**功能**:
- Hydra 配置管理
- W&B 日志记录 (在线/离线模式)
- 模型 checkpoint 保存
- 早停机制
- 学习率调度 (cosine warmup)
- 梯度裁剪

### 3.3 使用方法

```bash
# 使用 NequIP (默认)
python diffcsp/train_amorphous.py

# 使用 EGNN
python diffcsp/train_amorphous.py model=amorphous_flow_egnn

# 使用 SchNet  
python diffcsp/train_amorphous.py model=amorphous_flow_schnet

# 自定义实验名称
python diffcsp/train_amorphous.py expname=my-experiment

# 修改训练参数
python diffcsp/train_amorphous.py model.learning_rate=1e-3 data.datamodule.batch_size.train=64

# 离线模式 (不连接 W&B 服务器)
python diffcsp/train_amorphous.py logging.wandb.mode=offline

# Debug 模式 (快速验证)
python diffcsp/train_amorphous.py train.pl_trainer.fast_dev_run=true
```

### 3.4 W&B 日志

记录的指标:
- `train_loss` / `val_loss` - 主要损失
- `train/loss_x`, `train/loss_y`, `train/loss_z` - 分量损失
- `train/cosine_similarity`, `val/cosine_similarity` - 速度场相似度
- `train/pred_magnitude`, `train/target_magnitude` - 速度幅值
- 学习率曲线
- 模型梯度/参数 (可选)

---

## ✅ 阶段四: 生成与评估 (已完成)

### 4.1 生成脚本

**已创建文件**: `scripts/generate_amorphous.py`

功能:
- 条件生成 (指定冷却速率)
- 批量生成
- 保存为 LAMMPS/XYZ 格式
- ODE 积分方法: Euler / RK4

```bash
# 基础使用
python scripts/generate_amorphous.py \
    --checkpoint path/to/model.ckpt \
    --cooling_rate 100 \
    --num_samples 100 \
    --output_dir generated/

# 生成所有冷却速率
python scripts/generate_amorphous.py \
    --checkpoint path/to/model.ckpt \
    --all_rates \
    --num_samples 20

# 使用 RK4 积分
python scripts/generate_amorphous.py \
    --checkpoint path/to/model.ckpt \
    --method rk4 \
    --steps 100
```

### 4.2 评估指标

**已创建文件**: `scripts/evaluate_amorphous.py`

| 指标 | 描述 | 已实现 |
|------|------|--------|
| RDF | 径向分布函数 | ✅ |
| 键角分布 | C-C-C 键角 | ✅ |
| 配位数 | 平均近邻数/sp杂化分布 | ✅ |
| 环统计 | 3-8 元环分布 | ✅ |

```bash
# 评估生成样本并与真实数据对比
python scripts/evaluate_amorphous.py \
    --generated generated/samples/ \
    --reference data/amorphous_carbon/data/ \
    --output evaluation_results/
```

**输出文件**:
- `generated_metrics.json` - 生成样本指标
- `reference_metrics.json` - 真实样本指标
- `comparison.json` - 对比结果 (RDF MSE, MAE等)
- `comparison_plot.png` - 可视化对比图

### 4.3 可视化工具

**已创建文件**: `scripts/visualize_amorphous.py`

功能:
- 单结构 2D 可视化
- 多结构网格可视化
- 生成 vs 真实对比
- 按配位数着色 (sp=蓝, sp²=绿, sp³=红)

```bash
# 单结构可视化
python scripts/visualize_amorphous.py --input sample.data --output plot.png

# 多结构网格
python scripts/visualize_amorphous.py --input generated/ --output grid.png

# 对比可视化
python scripts/visualize_amorphous.py --generated gen.data --reference ref.data --output compare.png
```

---

## 📁 项目结构

```
CrystalFlow/
├── conf/
│   ├── amorphous_flow.yaml          # ✅ 主配置入口
│   ├── data/
│   │   └── amorphous_carbon.yaml    # ✅ 数据配置
│   ├── logging/
│   │   └── amorphous_flow.yaml      # ✅ W&B 日志配置
│   ├── train/
│   │   └── amorphous_flow.yaml      # ✅ 训练配置
│   └── model/
│       ├── amorphous_flow.yaml      # ✅ NequIP 配置
│       ├── amorphous_flow_egnn.yaml # ✅ EGNN 配置
│       └── amorphous_flow_schnet.yaml # ✅ SchNet 配置
├── data/
│   └── amorphous_carbon/
│       └── data/                    # LAMMPS 数据文件
├── diffcsp/
│   ├── train_amorphous.py           # ✅ 训练脚本
│   ├── pl_data/
│   │   └── amorphous_dataset.py     # ✅ 数据加载
│   └── pl_modules/
│       ├── nequip_flow.py           # ✅ NequIP 模型
│       ├── flow_transforms.py       # ✅ Flow Matching 变换
│       ├── amorphous_flow_module.py # ✅ Lightning 模块
│       └── model_factory.py         # ✅ 模型工厂
├── scripts/
│   ├── prepare_amorphous_carbon.py  # ✅ 数据预处理
│   ├── generate_amorphous.py        # ✅ 生成脚本
│   ├── evaluate_amorphous.py        # ✅ 评估脚本
│   └── visualize_amorphous.py       # ✅ 可视化脚本
├── generated/                       # 生成样本目录
└── evaluation_results/              # 评估结果目录
```

---

## 🔧 快速开始

### 安装依赖

```bash
conda activate crystalflow
pip install e3nn  # NequIP 需要
```

### 测试数据加载

```bash
cd CrystalFlow
python -c "
from diffcsp.pl_data.amorphous_dataset import AmorphousDataModule

dm = AmorphousDataModule(
    data_dir='data/amorphous_carbon',
    batch_size=4,
    auto_extract_rate=True,
)
dm.setup('fit')
print(f'Train: {len(dm.train_dataset)}, Val: {len(dm.val_dataset)}')
"
```

### 测试模型

```bash
python -c "
from diffcsp.pl_modules.model_factory import create_model, list_available_models
import torch

print('Available models:', list_available_models())

for model_name in ['nequip', 'egnn', 'schnet']:
    model = create_model(model_name)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{model_name}: {n_params:,} parameters')
"
```

### 训练 (待实现)

```bash
python diffcsp/run.py \
    data=amorphous_carbon \
    model=amorphous_flow \
    model.model_type=nequip \
    train.max_epochs=1000
```

---

## 📊 进度跟踪

| 阶段 | 状态 | 完成度 |
|------|------|--------|
| 阶段一: 数据处理 | ✅ 完成 | 100% |
| 阶段二: 网络改造 | ✅ 完成 | 100% |
| 阶段三: 训练集成 | ✅ 完成 | 100% |
| 阶段四: 生成评估 | ✅ 完成 | 100% |

---

## 🚀 端到端流程示例

### 1. 训练模型
```bash
cd CrystalFlow
source ~/miniconda3/bin/activate crystalflow

# 使用默认配置训练 NequIP
python diffcsp/train_amorphous.py expname=nequip-v1

# 或使用 EGNN (更快)
python diffcsp/train_amorphous.py model=amorphous_flow_egnn expname=egnn-v1
```

### 2. 生成样本
```bash
python scripts/generate_amorphous.py \
    --checkpoint hydra/singlerun/nequip-v1/epoch=XXX-val_loss=X.XX.ckpt \
    --all_rates \
    --num_samples 50
```

### 3. 评估质量
```bash
python scripts/evaluate_amorphous.py \
    --generated generated/YYYYMMDD_HHMMSS/ \
    --reference data/amorphous_carbon/data/ \
    --output evaluation_results/
```

### 4. 可视化
```bash
python scripts/visualize_amorphous.py \
    --input generated/YYYYMMDD_HHMMSS/rate_100 \
    --output visualization.png
```

---

## 🤝 与 AMC-FlowGen 的对比

| 功能 | 本项目 | AMC-FlowGen |
|------|--------|-------------|
| 模型后端 | NequIP/EGNN/SchNet | GNN/EGNN/SchNet |
| E3 等变 | ✅ (NequIP) | ❌ |
| 条件生成 | ✅ 冷却速率 | ✅ 冷却速率 |
| 框架 | PyTorch Lightning | PyTorch |
| 配置系统 | Hydra | 手动 |

**主要优势**:
1. NequIP 提供更强的几何等变性
2. Hydra 配置更灵活
3. Lightning 集成更完善
