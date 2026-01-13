# CrystalFlow → 非晶材料生成改造计划

## 核心策略

**保留 CrystalFlow 的 Flow Matching 框架** + **迁移 DM2 的非晶处理能力**

- Flow Matching (保留)
- Conditional CFG (保留)
- ODE求解器 (保留)
- Hydra配置 (保留)
- 非晶图构建 (迁移自DM2)
- EGNN/E3NN (迁移自DM2)
- 数据增强 (迁移自DM2)
- 周期性工具 (迁移自DM2)

---

## 阶段1: 数据层融合

**作用**: 让CrystalFlow能够读取和处理非晶材料数据  
**原因**: 晶体有周期性+对称性，非晶只有周期性边界。需要DM2的周期性工具处理无序结构

### 📦 从DM2直接复制
```bash
DM2/src/graphite/data/mol.py → CrystalFlow/diffcsp/pl_data/mol_data.py
DM2/src/graphite/nn/utils/mic.py → CrystalFlow/diffcsp/common/mic.py
DM2/src/graphite/nn/utils/periodic_radius_graph.py → CrystalFlow/diffcsp/common/periodic_radius_graph.py
DM2/src/graphite/nn/utils/edges.py → CrystalFlow/diffcsp/common/edges.py
```

### ✍️ 需要自己改写

**1.1 创建非晶数据集类**
- 文件: `diffcsp/pl_data/dataset.py`
- 新增: `AmorphousDataset(CrystDataset)`
- **作用**: 统一晶体/非晶数据格式，兼容现有训练流程  
- **原因**: 晶体用分数坐标+极坐标晶格，非晶用分数坐标+正交盒，需要转换层
- 功能:
  - 使用 MolData 替代 Data (DM2的数据结构，支持周期性)
  - 晶格简化为正交盒 (非晶不需要复杂晶格表示)
  - 加载退火速率条件 (控制生成结构的无序度)

**1.2 数据配置**
- 文件: `conf/data/amorphous_carbon.yaml`
- **作用**: 关闭晶体专用特性，启用非晶图构建  
- **原因**: 非晶无空间群/对称性，用物理截断半径代替晶体学方法
- 关键配置:
  - `niggli: false`, `primitive: false` (关闭晶胞约化)
  - `graph_method: radius_pbc` (固定截断半径，而非CrystalNN)
  - `cutoff_radius: 5.0`, `max_neighbors: 32` (碳材料典型值)
  - 条件: `annealing_rate` (对数尺度，跨10个数量级)

**1.3 数据准备脚本**
- 文件: `scripts/prepare_amorphous_data.py`
- **作用**: 从MD模拟轨迹提取训练数据  
- **原因**: 非晶结构来自分子动力学淬火，需要解析.xyz/.lammps格式

---

## 阶段2: 图构建层适配

**作用**: 构建原子间的连接关系（图的边）  
**原因**: 非晶无晶体学规则，用固定截断半径+最小镜像约定保证物理正确性

### 📦 DM2组件保持不变
- `periodic_radius_graph` 核心逻辑
- `minimum_image_convention` (MIC)
- 边特征计算 (`edge_vec`, `edge_length`)

### ✍️ 需要改写

**2.1 批处理图构建**
- 文件: `diffcsp/common/data_utils.py`
- 函数: `build_amorphous_batch_graph(batch, cutoff, max_neighbors)`
- **作用**: 将DM2的单样本图构建扩展为批量处理  
- **原因**: DM2只处理单个结构，训练需要批量并行加速
- 功能:
  - 遍历batch中每个图 (不同样本原子数不同)
  - 调用DM2的 `periodic_radius_graph` (核心算法)
  - 累加边索引偏移 (batch拼接后索引需要累加)

**2.2 DataModule对接**
- 文件: `diffcsp/pl_data/datamodule.py`
- **作用**: 根据数据类型自动选择图构建方法  
- **原因**: 晶体用CrystalNN，非晶用radius_pbc，需要兼容两种模式
- 修改 `collate_fn`: 检测数据类型分发图构建方法

---

## 阶段3: GNN架构混合

**作用**: 学习原子间相互作用，预测速度场  
**原因**: CrystalFlow的GemNet-dT依赖晶格对称性，非晶需要等变GNN捕捉无序结构

### 📦 迁移DM2的GNN模型

**选项A: EGNN (推荐)**
```bash
DM2/src/graphite/nn/models/egnn.py → CrystalFlow/diffcsp/pl_modules/egnn_model.py
DM2/src/graphite/nn/conv/egnn.py → CrystalFlow/diffcsp/pl_modules/conv/egnn_conv.py
```
- **作用**: 平移+旋转等变的消息传递网络  
- **原因**: 非晶无固定取向，等变性保证预测与坐标系无关，简单高效

**选项B: E3NN-NequIP (高精度)**
```bash
DM2/src/graphite/nn/models/e3nn_nequip_improved.py → CrystalFlow/diffcsp/pl_modules/e3nn_model.py
```

**径向基函数**
```bash
DM2/src/graphite/nn/basis.py → CrystalFlow/diffcsp/pl_modules/basis.py
```
- **作用**: 将原子间距离编码为高维特征  
- **原因**: 替代晶体的Miller指数，用Bessel函数捕捉距离依赖的相互作用

### ✍️ 需要改写

**3.1 非晶解码器**
- 文件: `diffcsp/pl_modules/amorphous_decoder.py`
- 类: `AmorphousDecoder(nn.Module)`
- **作用**: 整合DM2的GNN和CrystalFlow的Flow架构  
- **原因**: DM2预测噪声(Diffusion)，我们需要改为预测速度场(Flow)
- 架构:
  - backbone: EGNN/E3NN (从DM2迁移，负责消息传递)
  - time_embedding: 复用CrystalFlow的 `SinusoidalTimeEmbedding` (Flow时间步)
  - cond_embedding: 新增退火速率嵌入 (条件控制)
  - output_head: 预测速度场 (3D向量，而非DM2的noise)

**3.2 模型配置**
- 文件: `conf/model/decoder/egnn_amorphous.yaml`
- 参数: `hidden_dim: 256`, `num_layers: 8`

---

## 阶段4: Flow核心改造

**作用**: 实现Flow Matching训练和采样逻辑  
**原因**: DM2用扩散模型(DDPM)，我们保持Flow Matching框架，速度更快且确定性更强

### ✍️ 完全自己写 (DM2用Diffusion，无法迁移)

**4.1 非晶Flow模型**
- 文件: `diffcsp/pl_modules/amorphous_flow.py`
- 类: `AmorphousFlow(BaseModule)`
- 继承自: `diffcsp/pl_modules/flow.py` 的 `CSPFlow`
- **作用**: Flow Matching的核心训练和采样引擎  
- **原因**: 保持CrystalFlow的优势(快速ODE采样)，去除晶体专用组件
- 关键修改:
  - 移除 `lattice_polar` 模块 (非晶不需要学习晶格)
  - 添加 `orthogonal_lattice` (固定正交盒子，减少自由度)
  - 使用 `AmorphousDecoder` 替代 `CSPNet` (EGNN替代GemNet)
  - `forward`: Flow Matching训练逻辑
    - 线性插值: `pos_t = pos_0 + t*(pos_1 - pos_0)` (构建从噪声到真实的路径)
    - 动态构建图 (每次前向传播重新计算邻居)
    - 预测速度场 (学习从噪声→结构的流动)
    - 损失: `MSE(pred_velocity, target_velocity)` (速度匹配)
  - `sample`: ODE积分推理
    - 初始化随机坐标 (t=1, 盒子内均匀分布)
    - 每步重建图 (坐标变化导致邻居变化)
    - 应用PBC (保持原子在盒子内)

**4.2 晶格简化**
- 文件: `diffcsp/pl_modules/lattice_utils.py`
- 类: `OrthogonalLattice`
- **作用**: 简化晶格表示，降低模型复杂度  
- **原因**: 非晶MD通常用正交盒子，无需学习6自由度晶格参数
- 功能:
  - `from_lengths(Lx, Ly, Lz)` → 正交盒 (对角矩阵)
  - `sample_random()` → 随机初始化 (Flow的t=1状态)
  - `apply_pbc(pos, cell)` → 周期性边界 (坐标映射到[0, 1))

**4.3 模型配置**
- 文件: `conf/model/amorphous_flow.yaml`
- 关键配置:
  - `lattice_type: orthogonal`, `lattice_fixed: true`
  - `cutoff_radius: 5.0`, `rebuild_graph_every_step: true`
  - `cost_position: 10.0`, `cost_lattice: 0.0`

---

## 阶段5: 条件生成 + 训练优化

**作用**: 通过退火速率控制生成结构，增强训练稳定性  
**原因**: 不同退火速率产生不同无序度；数据增强防止过拟合

### 📦 迁移DM2数据增强
```bash
DM2/src/graphite/transforms/rattle_particles.py → CrystalFlow/diffcsp/pl_data/transforms/rattle.py
DM2/src/graphite/transforms/downselect_edges.py → CrystalFlow/diffcsp/pl_data/transforms/edge_dropout.py
```

### ✍️ 需要改写

**5.1 条件嵌入**
- 文件: `diffcsp/pl_modules/conditioning.py`
- 类: `AnnealingRateEmbedding(MultiEmbedding)`
- **作用**: 将退火速率编码为可学习的向量  

- 功能:
  - 对数归一化: `log10(rate)` → [0, 1] (线性化大范围)
  - MLP嵌入: [1] → [64] (学习条件表示)
  - 训练: CFG dropout (10%概率置零，学习有/无条件)
  - 推理: 混合有/无条件预测 (增强条件控制力度)

**5.2 数据增强集成**
- 文件: `diffcsp/pl_data/datamodule.py`
- **作用**: 训练时引入随机扰动，提高泛化能力  
- **原因**: 非晶本身就是无序的，增强数据多样性避免记忆训练集
- 修改 `train_dataloader`:
  - 添加 `RattleParticles(stdev=0.05)` (随机扰动坐标±0.05Å)
  - 添加 `DownselectEdges(keep_ratio=0.9)` (随机删除10%的边)

**5.3 训练脚本**
- 文件: `scripts/train_amorphous.sh`
- 核心参数:
  - `data=amorphous_carbon`
  - `model=amorphous_flow`
  - `model.cutoff_radius=5.0`
  - `+model.guide_threshold=-1`
  - `+train.pl_trainer.gradient_clip_val=1.0`

---

## 文件清单

### 📦 从DM2直接复制 (7个文件)
| DM2源文件 | CrystalFlow目标 | 修改 |
|----------|----------------|------|
| `data/mol.py` | `pl_data/mol_data.py` | ❌ 无 |
| `nn/utils/mic.py` | `common/mic.py` | ❌ 无 |
| `nn/utils/periodic_radius_graph.py` | `common/periodic_radius_graph.py` | ❌ 无 |
| `nn/utils/edges.py` | `common/edges.py` | ❌ 无 |
| `nn/models/egnn.py` | `pl_modules/egnn_model.py` | ✅ 输出层 |
| `nn/basis.py` | `pl_modules/basis.py` | ❌ 无 |
| `transforms/rattle_particles.py` | `pl_data/transforms/rattle.py` | ✅ Hydra集成 |

### ✍️ 需要自己写 (5个核心文件)
| 文件 | 内容 | 难度 |
|------|------|------|
| `pl_data/dataset.py` | `AmorphousDataset` | ⭐⭐ |
| `common/data_utils.py` | 批量图构建 | ⭐⭐⭐ |
| `pl_modules/amorphous_decoder.py` | Flow解码器 | ⭐⭐⭐⭐ |
| `pl_modules/amorphous_flow.py` | Flow主模型 | ⭐⭐⭐⭐⭐ |
| `pl_modules/lattice_utils.py` | 正交盒工具 | ⭐ |

### 🔧 需要修改的现有文件
- `pl_data/datamodule.py`: 添加非晶数据支持
- `pl_modules/conditioning.py`: 新增退火速率嵌入
- `run.py`: 检测模型类型分发

---

## 关键差异处理

| 项目 | CrystalFlow | DM2 | 统一方案 |
|------|------------|-----|---------|
| **坐标** | `frac_coords` (分数) | `pos` (笛卡尔) | Dataset转换时统一为笛卡尔 |
| **晶格** | `lattice_polar` (6D) | `cell` (3×3) | 简化为正交盒 `diag([Lx,Ly,Lz])` |
| **图方法** | `CrystalNN` (动态) | `radius` (固定) | 配置项 `graph_method` |
| **GNN** | GemNet-dT (晶体) | EGNN (等变) | 新增 `AmorphousDecoder` |

---

## 验证检查点

### Milestone 1: 数据通路
```python
dataset = AmorphousDataset(...)
batch = dataset[0]
assert isinstance(batch, MolData)  # ✅ DM2结构
assert batch.pos.shape[-1] == 3    # ✅ 笛卡尔坐标
```

### Milestone 2: 图构建
```python
edge_index, edge_attr = build_amorphous_batch_graph(batch, cutoff=5.0)
assert 'edge_vec' in edge_attr  # ✅ DM2特征
```

### Milestone 3: 模型训练
```python
model = AmorphousFlow(...)
loss = model.training_step(batch)
assert loss < 10.0 and not torch.isnan(loss)  # ✅ 收敛
```

### Milestone 4: 生成质量
```python
structures = model.sample(num_samples=10, annealing_rate=1e12)
rdf_error = compute_rdf_error(structures, reference)
assert rdf_error < 0.05  # ✅ RDF误差<5%
```

---

## 优先级

1. **P0** (必须): 数据层 + 图构建 + Flow核心
2. **P1** (重要): EGNN迁移 + 条件生成  
3. **P2** (优化): 数据增强 + 评估指标

## 技术选型

- **GNN**: EGNN (简单稳定)
- **晶格**: 固定正交盒 (减少自由度)
- **采样**: Euler ODE (已验证)

## 调试策略

1. 先用小数据集(100样本)验证流程
2. 固定晶格，只学习坐标
3. 无条件训练通过后再加CFG
