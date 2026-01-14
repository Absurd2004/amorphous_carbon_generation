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

## 🔄 阶段三: 训练框架集成 (进行中)

### 3.1 配置文件

**创建文件**: `conf/model/amorphous_flow.yaml`

```yaml
_target_: diffcsp.pl_modules.amorphous_flow_module.AmorphousFlowModule

# Model selection
model_type: nequip  # 'nequip', 'egnn', 'schnet'

# Model configuration (model-specific)
model_config:
  # NequIP specific
  irreps_hidden: '64x0e + 32x1e + 32x2e'
  num_convs: 4
  radial_neurons: [32, 64]
  
# Common configuration  
cutoff: 5.0
time_embed_dim: 32
cond_embed_dim: 32
cond_min_value: 1.0  # log10(10)
cond_max_value: 4.5  # log10(30000)

# Training configuration
box_size: [12.0, 12.0, 20.0]
is_2d: true
prior: uniform
use_condition: true

# Optimizer
learning_rate: 1e-4
weight_decay: 0.0
```

### 3.2 数据配置

**创建文件**: `conf/data/amorphous_carbon.yaml`

```yaml
_target_: diffcsp.pl_data.amorphous_dataset.AmorphousDataModule

data_dir: ${paths.data_dir}/amorphous_carbon
cutoff: 5.0
duplicate: 128  # 数据增强
train_ratio: 0.8
val_ratio: 0.1
batch_size: 32
num_workers: 4
auto_extract_rate: true
```

### 3.3 训练脚本

**任务清单**:
- [ ] 创建 Hydra 配置文件
- [ ] 修改 `run.py` 支持 amorphous flow
- [ ] 添加 checkpoint 回调
- [ ] 添加 TensorBoard 日志
- [ ] 添加早停机制

---

## 📋 阶段四: 生成与评估 (待开始)

### 4.1 生成脚本

**创建文件**: `scripts/generate_amorphous.py`

功能:
- 条件生成 (指定冷却速率)
- 批量生成
- 保存为 LAMMPS 格式

```python
# 使用示例
python scripts/generate_amorphous.py \
    --checkpoint path/to/model.ckpt \
    --cooling_rate 100 \
    --num_samples 100 \
    --output_dir generated/
```

### 4.2 评估指标

| 指标 | 描述 |
|------|------|
| RDF | 径向分布函数 |
| 键角分布 | C-C-C 键角 |
| 配位数 | 平均近邻数 |
| 环统计 | 3-8 元环分布 |
| 能量 | LAMMPS/ASE 计算 |

### 4.3 可视化

- 结构可视化 (ASE/OVITO)
- 训练曲线
- 条件插值

---

## 📁 项目结构

```
CrystalFlow/
├── conf/
│   ├── data/
│   │   └── amorphous_carbon.yaml    # 数据配置
│   └── model/
│       └── amorphous_flow.yaml      # 模型配置
├── data/
│   └── amorphous_carbon/
│       └── data/                    # LAMMPS 数据文件
├── diffcsp/
│   ├── pl_data/
│   │   └── amorphous_dataset.py     # ✅ 数据加载
│   └── pl_modules/
│       ├── nequip_flow.py           # ✅ NequIP 模型
│       ├── flow_transforms.py       # ✅ Flow Matching 变换
│       ├── amorphous_flow_module.py # ✅ Lightning 模块
│       └── model_factory.py         # ✅ 模型工厂
└── scripts/
    ├── prepare_amorphous_carbon.py  # 数据预处理
    └── generate_amorphous.py        # 生成脚本 (待创建)
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
| 阶段三: 训练集成 | 🔄 进行中 | 30% |
| 阶段四: 生成评估 | ⏳ 待开始 | 0% |

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
