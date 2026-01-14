# 2D 非晶碳 Flow Matching 生成模型 - 实施计划

## 项目概述

**目标**: 基于 DM2 的 NequIP 等变神经网络架构，改造为 Flow Matching 生成模型，专为 2D 非晶碳条件生成。

**数据特点**:
- LAMMPS 格式文件 (`.data`)
- 50 个碳原子 / 样本
- 2D 结构 (z = 0)
- 固定盒子: 12×12×20 Å
- 约 1000 个样本
- 条件: 退火速率 = **100 K/ps** (暂时固定，后续可扩展)
- 数据划分: **train:val:test = 8:1:1**

**技术路线**: DM2 NequIP 架构 + Flow Matching 训练逻辑 + CrystalFlow 框架集成

---

## ✅ 阶段一: 环境准备与数据处理 (已完成)

**已创建文件**:
- `diffcsp/pl_data/amorphous_dataset.py` - 数据加载模块
- `scripts/prepare_amorphous_carbon.py` - 数据预处理脚本

**已验证**: 1000个样本加载成功，图结构正确，2D约束有效

---

## ✅ 阶段二: Flow Matching 网络改造 (已完成)

**已创建文件**:
- `diffcsp/pl_modules/nequip_flow.py` - NequIP_FlowMatching 网络
- `diffcsp/pl_modules/flow_transforms.py` - Flow Matching 变换
- `diffcsp/pl_modules/amorphous_flow_module.py` - Lightning 训练模块

**核心组件**:
1. **NequIP_FlowMatching**: E3等变网络，支持时间嵌入和条件嵌入
2. **FlowMatchingTransform**: 实现 x_t = (1-t)x_0 + t*x_1 插值
3. **AmorphousFlowModule**: 完整的 Lightning 模块，包含训练和采样

**已验证**:
- 前向传播成功 ✓
- 损失计算正确 ✓
- 反向传播正常 ✓
- 采样生成有效 ✓

---

## 阶段一: 环境准备与数据处理 (预计 2-3 天)

### 1.1 复用 DM2 数据加载代码

**文件来源**: `DM2/demo/demo_training/denoiser_train_unconditional.py`

**需要复用的函数**:
```python
# 1. ASE 图构建函数
def ase_graph(data, cutoff):
    i, j, D = primitive_neighbor_list('ijD', cutoff=cutoff, pbc=data.pbc, 
                                      cell=data.cell, positions=data.pos.numpy(), 
                                      numbers=data.numbers)
    data.edge_index = torch.tensor(np.stack((i, j)), dtype=torch.long)
    data.edge_attr = torch.tensor(D, dtype=torch.float)
    return data

# 2. 数据集类 (需修改以支持条件)
class PeriodicStructureDataset(Dataset):
    ...
```

**创建新文件**: `CrystalFlow/diffcsp/pl_data/amorphous_dataset.py`

**任务清单**:
- [ ] 从 DM2 复制 `ase_graph` 函数
- [ ] 创建 `AmorphousCarbonDataset` 类，支持:
  - [x] LAMMPS 文件读取
  - [ ] 退火速率条件解析 (从文件名提取)
  - [ ] 笛卡尔坐标归一化
  - [ ] 数据复制增强
- [ ] 创建 `AmorphousDataModule` 类

**测试点**:
```bash
# 测试数据加载
 **主线修订**: 基于 DM2 的 NequIP 等变神经网络架构，改造为 Flow Matching 生成模型，专为 2D 非晶碳条件生成。
from diffcsp.pl_data.amorphous_dataset import AmorphousCarbonDataset
ds = AmorphousCarbonDataset('/path/to/data', cutoff=5.0)

 **技术路线修订**: DM2 NequIP 架构 + Flow Matching 训练逻辑 + DM2 数据处理 + 条件嵌入
- 读取所有 LAMMPS 文件
- 划分 train/val/test (80/10/10)
- 保存为 PyTorch 缓存文件 (`.pt`)

```
# 退火速率: 统一设为 100 K/ps
# 后续扩展: 可改为 rate_100_sample_001.data 格式
```
- 训练集: 800 个样本 (1-800)
- 验证集: 100 个样本 (801-900)
- 测试集: 100 个样本 (901-1000)
```

## 阶段二: 模型架构简化 (预计 3-5 天)
### 2.1 创建 AmorphousFlow 模型

**创建文件**: `CrystalFlow/diffcsp/pl_modules/amorphous_flow.py`
**主要简化**:
| 原 CSPFlow 功能 | AmorphousFlow 处理 |
|----------------|-------------------|
| 晶格流 (lattice flow) | ❌ 移除 |
| 坐标流 | ✅ 保留并适配 2D |
**核心修改**:

```python
        self.keep_lattice = True  # 始终固定盒子
        
        # 2D 特殊处理
        self.is_2d = True  # z 坐标始终为 0
        
    def forward(self, batch):
        # 只进行坐标流
        cart_coords = batch.pos  # 笛卡尔坐标
        box_size = batch.cell.diagonal()  # 盒子尺寸
        
        # 归一化到 [0, 1]
        norm_coords = cart_coords / box_size
        
        # 采样初始坐标 (均匀分布)
        x0 = torch.rand_like(norm_coords)
        if self.is_2d:
            x0[:, 2] = 0  # z = 0
        
        # Flow matching
        tar_x = (norm_coords - x0 - 0.5) % 1 - 0.5  # 最小镜像
        input_coords = x0 + times * tar_x
        # 预测
        pred_x = self.decoder(...)
        
        # 损失 (只有坐标)
        loss = F.mse_loss(pred_x, tar_x)
        return {'loss': loss}
```

**任务清单**:
- [ ] 创建 `AmorphousFlow` 类框架
- [ ] 实现简化的 `forward()` 方法
- [ ] 实现 `sample()` 采样方法
- [ ] 添加 2D 坐标约束
**测试点**:
```python
# 测试前向传播
model = AmorphousFlow(...)
batch = next(iter(dataloader))
loss = model(batch)
print(f'Loss: {loss["loss"].item():.4f}')
```


**修改文件**: `CrystalFlow/diffcsp/pl_modules/cspnet.py`

```python
    if self.edge_style == 'knn_pbc':
        # 使用 DM2 的 primitive_neighbor_list
        ...
```

2. **移除晶格特征**:
```python
# 禁用
rec_emb = None      # 倒易晶格嵌入
periodic_norm = False
use_angles = False
ip = False          # 晶格内积
```

3. **输出调整**:
```python
# 只输出坐标变化
def forward(self, ...):
    ...
    coord_out = self.coord_out(node_features)
    # 移除 lattice_out
    return coord_out
```

**任务清单**:
- [ ] 添加 `edge_style='knn_pbc'` 模式 (使用 ASE)
- [ ] 创建简化配置选项
- [ ] 测试非周期图构建正确性

---

## 阶段三: 条件嵌入集成 (预计 2-3 天)

### 3.1 从 DM2 迁移条件嵌入

**源文件**: `DM2/src/graphite/nn/models/e3nn_nequip.py`

**复制到**: `CrystalFlow/diffcsp/pl_modules/conditioning.py` (已有部分代码)

**需要迁移**:

```python
# GaussianBasisEmbedding - 高斯基函数嵌入
class GaussianBasisEmbedding(nn.Module):
    def __init__(self, num_basis=12, embedding_dim=32, min_value=-1, max_value=3):
        # min_value=-1, max_value=3 适合 log10(cooling_rate)
        ...
    
    def forward(self, x):
        # x: [batch_size] - log10 cooling rate
        return embedding  # [batch_size, embedding_dim]
```

### 3.2 条件注入方式

**方案A**: 加到节点特征 (类似 DM2)
```python
# 在每层后添加条件
for layer in self.layers:
    h = layer(h, ...)
    h = h + condition_embed  # 广播到所有节点
```

**方案B**: 使用 CrystalFlow 已有的 `guide_threshold` 机制
```python
# flow.py 已有条件嵌入接口
cemb = self.cond_emb(**{key: batch.get(key) for key in self.cond_emb.cond_keys})
```

**推荐方案A** - 更简单直接

**任务清单**:
- [ ] 迁移 `GaussianBasisEmbedding` 类
- [ ] 在 decoder 中添加条件注入点
- [ ] 实现条件采样 (classifier-free guidance 可选)

---

## 阶段四: 配置与训练脚本 (预计 2-3 天)

### 4.1 创建配置文件

**创建文件**: `CrystalFlow/conf/data/amorphous_carbon.yaml`
```yaml
root_path: ${oc.env:PROJECT_ROOT}/data/amorphous_carbon
prop: cooling_rate
num_targets: 1
properties:
  - cooling_rate
conditions:
  - cooling_rate

# 2D 非晶碳特有
num_atoms: 50
box_size: [12.0, 12.0, 20.0]
is_2d: true
cutoff: 5.0

# 数据增强
duplicate: 128

datamodule:
  _target_: diffcsp.pl_data.amorphous_dataset.AmorphousDataModule
  datasets:
    train:
      _target_: diffcsp.pl_data.amorphous_dataset.AmorphousCarbonDataset
      ...
```

**创建文件**: `CrystalFlow/conf/model/amorphous_flow.yaml`
```yaml
_target_: diffcsp.pl_modules.amorphous_flow.AmorphousFlow

time_dim: 256
latent_dim: 0
cost_coord: 1.0
cost_lattice: 0.0  # 禁用

is_2d: true
use_pbc: true

decoder:
  _target_: diffcsp.pl_modules.amorphous_cspnet.AmorphousCSPNet
  hidden_dim: 128
  num_layers: 4
  edge_style: 'knn_pbc'
  cutoff: 5.0
  max_neighbors: 20
  dis_emb: 'sin'
  rec_emb: null  # 禁用
  periodic_norm: false
  
conditions:
  embedding_dim: 32
  num_basis: 12
```

### 4.2 训练脚本

**运行命令**:
```bash
CUDA_VISIBLE_DEVICES=0 python diffcsp/run.py \
  data=amorphous_carbon \
  model=amorphous_flow \
  data.train_max_epochs=3000 \
  optim.optimizer.lr=1e-3 \
  expname=amorphous-carbon-2d
```

**任务清单**:
- [ ] 创建数据配置文件
- [ ] 创建模型配置文件
- [ ] 验证配置加载正确
- [ ] 小规模训练测试 (fast_dev_run)

---

## 阶段五: 生成与评估 (预计 3-5 天)

### 5.1 创建生成脚本

**创建文件**: `CrystalFlow/scripts/generate_amorphous.py`

**参考**: `DM2/demo/demo_generating/denoise_generate_unconditional.py`

**核心函数**:
```python
@torch.no_grad()
def generate_amorphous(model, num_atoms, box_size, steps=100, condition=None):
    """
    从随机噪声生成 2D 非晶碳结构
    """
    # 初始化随机坐标
    x0 = torch.rand(num_atoms, 3) * box_size
    x0[:, 2] = 0  # 2D
    
    # ODE 求解 (使用 CrystalFlow 的 solver)
    for t in linspace(0, 1, steps):
        v = model.decoder(x0, t, condition)
        x0 = x0 + v * dt
        x0 = x0 % box_size  # PBC
    
    return x0
```

### 5.2 评估指标 (参考 DM2)

**DM2 使用的评估方法**:

1. **径向分布函数 (RDF)** - 最重要的结构指标
   - 计算原子对距离分布
   - 对比生成结构与真实结构的 RDF 曲线
   - 使用 Jensen-Shannon 散度量化差异

2. **键角分布 (Bond Angle Distribution)**
   - 来源: `DM2/src/graphite/nn/utils/angles.py`
   - 计算相邻原子间的键角 cos/sin 值
   - 对于碳材料，典型键角约 120° (sp2) 或 109.5° (sp3)

3. **二面角分布 (Dihedral Angles)** - 可选
   - 四原子链的扭转角
   - 反映结构的三维特征 (2D 碳可能不适用)

4. **Chamfer 距离**
   - 来源: `DM2/src/graphite/nn/loss.py`
   - 点云之间的最近邻距离
   - 可用于评估生成结构与参考结构的相似度

5. **配位数分布**
   - 每个原子的邻居数量统计
   - 对于 2D 碳，典型配位数为 2-3

**复用 DM2 代码**:
```python
# 从 DM2 迁移
from graphite.nn.loss import jensen_shannon, chamfer_distance
from graphite.nn.utils.angles import bond_angles
```

**创建文件**: `CrystalFlow/scripts/evaluate_amorphous.py`

**任务清单**:
- [ ] 实现生成脚本
- [ ] 实现 RDF 计算
- [ ] 实现配位数分析
- [ ] 可视化工具

---

## 阶段六: 优化与扩展 (可选)

### 6.1 性能优化
- [ ] 多 GPU 训练 (DDP)
- [ ] 混合精度训练
- [ ] 数据加载优化

### 6.2 模型改进
- [ ] 尝试 E3NN (从 DM2 迁移)
- [ ] Classifier-free guidance
- [ ] 不同采样方法 (Euler, RK4, DPM-Solver)

### 6.3 扩展到 3D
- [ ] 移除 z=0 约束
- [ ] 调整盒子尺寸
- [ ] 3D 非晶碳数据集

---

## 文件结构规划

```
CrystalFlow/
├── conf/
│   ├── data/
│   │   └── amorphous_carbon.yaml       # [新建]
│   └── model/
│       └── amorphous_flow.yaml         # [新建]
├── data/
│   └── amorphous_carbon/
│       ├── data/                       # [已有] LAMMPS 文件
│       ├── train.pt                    # [待生成] 训练集缓存
│       ├── val.pt                      # [待生成] 验证集缓存
│       └── test.pt                     # [待生成] 测试集缓存
├── diffcsp/
│   ├── pl_data/
│   │   └── amorphous_dataset.py        # [新建] 数据集类
│   └── pl_modules/
│       ├── amorphous_flow.py           # [新建] 流模型
│       └── amorphous_cspnet.py         # [新建] Decoder (可选，或修改原 cspnet.py)
└── scripts/
    ├── prepare_amorphous_carbon.py     # [新建] 数据预处理
    ├── generate_amorphous.py           # [新建] 生成脚本
    └── evaluate_amorphous.py           # [新建] 评估脚本
```

---

## 里程碑与时间表

| 阶段 | 内容 | 预计时间 | 测试验证 |
|------|------|----------|----------|
| 1 | 数据加载 | 2-3 天 | 数据集可迭代，图结构正确 |
| 2 | 模型架构 | 3-5 天 | 前向传播无报错，loss 下降 |
| 3 | 条件嵌入 | 2-3 天 | 条件生成结果有差异 |
| 4 | 配置与训练 | 2-3 天 | 完整训练流程跑通 |
| 5 | 生成与评估 | 3-5 天 | RDF 与真实数据接近 |
| 6 | 优化扩展 | 持续 | 性能/效果提升 |

**总计**: 约 2-3 周完成基础版本

---

## 风险与注意事项

1. **2D 结构特殊性**: 确保 z 坐标始终为 0
2. **PBC 处理**: 使用 ASE 的 `primitive_neighbor_list` 确保正确
3. **坐标归一化**: 注意盒子尺寸不一致时的处理
4. **条件标注**: 需要用户提供退火速率的标注方式
5. **小数据量**: 1000 个样本可能需要更多数据增强

---

## 待用户确认

~~1. **退火速率标注**: 请确认如何在文件名或元数据中体现退火速率~~
   - ✅ 已确认：暂时统一设为 100 K/ps

~~2. **数据划分**: 是否有特定的 train/val/test 划分要求？~~
   - ✅ 已确认：8:1:1 比例

~~3. **评估指标**: 除 RDF 外，还需要哪些评估指标？~~
   - ✅ 已确认：参考 DM2 (RDF, 键角分布, Chamfer 距离, 配位数)

4. **硬件资源**: 可用 GPU 型号和数量？

5. **时间约束**: 是否有截止日期？

---

## 下一步行动

1. ~~等待用户确认退火速率标注方式~~ ✅
2. 开始阶段一：数据加载模块开发
3. 创建 `amorphous_dataset.py` 文件
4. 测试数据加载和图构建
